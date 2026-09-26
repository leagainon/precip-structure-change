import pandas as pd
import pyarrow.dataset as ds

##### Define a lat/lon window
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
    (meta_data.station_coordinates_wgs84_lat >= 46.3)
    & (meta_data.station_coordinates_wgs84_lat < 47.3)
    & (meta_data.station_coordinates_wgs84_lon >= 7.2)
    & (meta_data.station_coordinates_wgs84_lon < 7.6)
)

stations = meta_data.loc[mask, "station_abbr"].tolist()

print("stations to load:", stations)

##### Load the data
dataset = ds.dataset(
    "/home/lea/Documents/Data/meteoswiss-weather-stations/partitioned_parquets/",
    partitioning="hive",
)

df = dataset.to_table(filter=ds.field("station_abbr").isin(stations)).to_pandas()

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

#### Compute the 2D variograms of the fields
