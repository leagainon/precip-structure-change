from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from src.precip_gstat import data_utils
from src.precip_gstat.config_reader import read_config
from src.precip_gstat.constants import FMT
from src.precip_gstat.variogram import compute_2d_variogram
from src.precip_gstat.viz import plot_2d_variogram

# ==========================
# Load the metadata and the data
# ==========================
config = read_config("config.toml")
df, meta_data = data_utils.load_station_data(config)

print("loaded the data")

# ==========================
# Select an event
# ==========================
date = pd.to_datetime("05.02.2017 06:00", format=FMT)
df_event = df[
    (df["reference_timestamp"] >= date)
    & (df["reference_timestamp"] <= date + pd.Timedelta(hours=16))
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
# Compute vario and plot per timestep
# =============================
timestamps = sorted(gdf["reference_timestamp"].unique())

for ts in timestamps:
    gdf_ts = gdf[gdf["reference_timestamp"] == ts].copy()

    # Skip timesteps where any station records zero — avoids zero-inflation bias
    # (Deutsch & Journel 1998). Remove once we do indicator variograms.
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
