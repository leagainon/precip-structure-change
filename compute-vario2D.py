from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from src.precip_gstat import data_utils
from src.precip_gstat.config_reader import read_config
from src.precip_gstat.constants import FMT
from src.precip_gstat.variogram import iter_2d_vario

# ==========================
# Load the metadata and the data
# ==========================
config = read_config("config.toml")
df, meta_data = data_utils.load_station_data(config)

# ==========================
# Select an event
# ==========================
date = pd.to_datetime("05.02.2017 06:00", format=FMT)
df_event = df[(df["t"] >= date) & (df["t"] <= date + pd.Timedelta(hours=16))]

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
ctx.add_basemap(ax=ax, source=ctx.providers.Esri.WorldGrayCanvas)  # pyright: ignore[] # comment this line if no internet access ow fatal error

ax.set_axis_off()

out = Path("output/tests_event2017-02-05/fig_stations.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=500)
# plt.show()


# =============================
# Compute vario and plot per timestep
# =============================
iter_2d_vario(config=config, gdf=gdf)
