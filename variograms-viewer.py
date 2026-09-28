from itertools import combinations
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

# ============================
# Define a lat/lon window
# ============================

meta_data_precip = pd.read_csv(
    "/home/lea/Documents/Data/meteoswiss-weather-stations/ogd-smn-precip_meta_stations.csv",
    sep=";",
    encoding="windows-1252",
)
meta_data = pd.read_csv(
    "/home/lea/Documents/Data/meteoswiss-weather-stations/ogd-smn_meta_stations.csv",
    sep=";",
    encoding="windows-1252",
)
meta_data = pd.concat([meta_data_precip, meta_data]).reset_index(drop=True)

mask = (
    (meta_data.station_coordinates_wgs84_lat >= 46.5)
    & (meta_data.station_coordinates_wgs84_lat < 47.5)
    & (meta_data.station_coordinates_wgs84_lon >= 6.0)
    & (meta_data.station_coordinates_wgs84_lon < 7.9)
)

stations = meta_data.loc[mask, "station_abbr"].tolist()

# ============================
# Load the data
# ============================
# 1. read the dataset file architecture
dataset = ds.dataset(
    "/home/lea/Documents/Data/meteoswiss-weather-stations/partitioned_parquets/",
    partitioning="hive",
)  #  hive very important here since we partiotioned the dataset

# 2. load only the data that we want
df = dataset.to_table(filter=ds.field("station_abbr").isin(stations)).to_pandas()

# 3. add the coordinates of the stations to the dataframe
df = df.merge(
    meta_data[
        [
            "station_abbr",
            "station_coordinates_wgs84_lon",
            "station_coordinates_wgs84_lat",
        ]
    ],
    on="station_abbr",
    how="left",
).rename(
    columns={
        "station_coordinates_wgs84_lat": "lat",
        "station_coordinates_wgs84_lon": "lon",
    }
)
# 4. transform the date strings in datetime objects
fmt = "%d.%m.%Y %H:%M"
df["reference_timestamp"] = pd.to_datetime(df.reference_timestamp, format=fmt)

# ==========================
# Select an event
# ==========================

date = pd.to_datetime("05.02.2017 06:00", format=fmt)
df_event = df[
    (df.reference_timestamp >= date)
    & (df.reference_timestamp <= date + pd.Timedelta(hours=16))
]

# =============================
# Build GeoDataFrame (metric CRS)
# =============================

gdf = gpd.GeoDataFrame(
    df_event,
    geometry=gpd.points_from_xy(
        df_event["lon"], df_event["lat"], crs="EPSG:4326"
    ).to_crs("EPSG:3857"),
)

# =============================
# Visualization of the data
# =============================

fig, ax = plt.subplots(figsize=(13, 8))
gdf.plot(
    ax=ax,
    column="rre150h0",
    cmap="Reds",
    markersize=50,
    legend=True,
    legend_kwds={"label": "Mean hourly precipitation (mm)"},
)
ctx.add_basemap(ax=ax, source=ctx.providers.Esri.WorldGrayCanvas)  # pyright: ignore[]

ax.set_axis_off()

out = Path("output/tests_event2017-02-05/fig_stations.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=500)
plt.show()

# =============================
# 2D Variogram functions
# =============================


def compute_2d_variogram(gdf_ts, value_col="rre150h0", n_bins=4, max_lag=None):
    """
    Compute 2D experimental variogram surface for a single timestep.

    Parameters
    ----------
    gdf_ts   : **GeoDataFrame** with a geom in EPSG:3857
    value_col: str of the precipitation column (mm)
    n_bins   : int, number of bins per half-axis. The total grid is (2*n_bins+1)^2
    max_lag  : float, max lag in meters. Default: 1/2 of the max pairwise distance

    Returns
    -------
    gamma     : (2*n_bins+1, 2*n_bins+1) array, mean semivariance (mm^2)
    counts    : same shape, number of pairs per bin
    bin_edges : lag bin edges in meters
    """
    # drop nans
    gdf_ts = gdf_ts[gdf_ts[value_col].notna()]

    coords = np.array([[geom.x, geom.y] for geom in gdf_ts.geometry])
    values = gdf_ts[value_col].values

    pairs = list(
        combinations(range(len(gdf_ts)), 2)
    )  # all combinations of the indices of the gdf
    i_idx, j_idx = np.array(pairs).T  # fomatted in two arrays for better handling

    # Forward lags
    dx_fwd = coords[j_idx, 0] - coords[i_idx, 0]
    dy_fwd = coords[j_idx, 1] - coords[i_idx, 1]
    sv = 0.5 * (values[j_idx] - values[i_idx]) ** 2

    # Mirror lags: same semivariance since it is just the same lag but in the opposite direction
    dx = np.concatenate([dx_fwd, -dx_fwd])
    dy = np.concatenate([dy_fwd, -dy_fwd])
    sq_diff = np.concatenate([sv, sv])

    # Define the maxlag if not provided
    if max_lag is None:
        max_lag = 0.5 * np.sqrt(dx_fwd**2 + dy_fwd**2).max()

    # Define the edges
    bin_edges = np.linspace(-max_lag, max_lag, 2 * n_bins + 1)
    n_cells = 2 * n_bins

    # Compute the semivariances
    gamma = np.full((n_cells, n_cells), np.nan)
    counts = np.zeros((n_cells, n_cells), dtype=int)
    for bi in range(n_cells):
        for bj in range(n_cells):
            x0, x1 = bin_edges[bi], bin_edges[bi + 1]  # bin of the dx of the lag vector
            y0, y1 = bin_edges[bj], bin_edges[bj + 1]  # bin of the dy of the lag vector
            mask = (
                (dx >= x0) & (dx < x1) & (dy >= y0) & (dy < y1)
            )  # only keep the lag vectors that fall in the specified bin
            if mask.sum() > 0:
                gamma[bi, bj] = sq_diff[
                    mask
                ].mean()  # semivariance in this bin = mean square diff of the correspinding pair values
                counts[bi, bj] = (
                    mask.sum()
                )  # also compute the number of paris in this bin

    return gamma, counts, bin_edges


def plot_2d_variogram(
    gamma, counts, bin_edges, title="2D Variogram", min_pairs=2, fs=16
):
    plt.rcParams.update({"font.size": 16})

    gamma_masked = np.where(counts >= min_pairs, gamma, np.nan)

    # Use edges directly with shading="flat" so that pcolormesh expects coordinate arrays of length n+1 for an (n,n) data array
    edges_km = bin_edges / 1000  # m to km, length = 2*n_bins+1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    im0 = axes[0].pcolormesh(
        edges_km, edges_km, gamma_masked.T, cmap="RdYlGn_r", shading="flat"
    )
    plt.colorbar(im0, ax=axes[0], label=r"Semivariance (mm$^2$)")
    axes[0].set_xlabel("Lag Easting (km)")
    axes[0].set_ylabel("Lag Northing (km)")
    axes[0].set_title(title, fontsize=fs)
    axes[0].axhline(0, color="k", lw=0.5, ls="--")
    axes[0].axvline(0, color="k", lw=0.5, ls="--")
    axes[0].set_aspect("equal")

    im1 = axes[1].pcolormesh(edges_km, edges_km, counts.T, cmap="Blues", shading="flat")
    plt.colorbar(im1, ax=axes[1], label="Pair count")
    axes[1].set_xlabel("Lag Easting (km)")
    axes[1].set_ylabel("Lag Northing (km)")
    axes[1].set_title("pair counts")
    axes[1].axhline(0, color="k", lw=0.5, ls="--")
    axes[1].axvline(0, color="k", lw=0.5, ls="--")
    axes[1].set_aspect("equal")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


# =============================
# Compute and plot per timestep
# =============================

timestamps = sorted(gdf["reference_timestamp"].unique())

for ts in timestamps:
    gdf_ts = gdf[gdf["reference_timestamp"] == ts].copy()

    # Skip timesteps where any station records zero — avoids zero-inflation bias
    # (Deutsch & Journel 1998). Remove once you move to indicator variograms.
    if (gdf_ts["rre150h0"] == 0).any():
        print(f"Skipping {ts}: zero precipitation at ≥1 station")
        continue

    if len(gdf_ts) < 5:
        print(f"Skipping {ts}: too few stations ({len(gdf_ts)})")
        continue

    gamma, counts, bin_edges = compute_2d_variogram(
        gdf_ts, value_col="rre150h0", n_bins=4
    )
    fig = plot_2d_variogram(gamma, counts, bin_edges, title=str(ts), min_pairs=2)

    out = Path("output/tests_event2017-02-05") / f"vario_{ts.strftime('%H')}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=500)
    plt.show()

