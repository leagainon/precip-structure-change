import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from .constants import FMT


def load_station_data(configuration) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    A data loader for meteoswiss stations data and metadata
    Inputs:
    ---
    - config : a config dict containing the data loading options
    Outputs:
    ---
    - df : pandas dataframe containing the stations data properly formatted
    - metadata : pandas dataframe of the stations metadata
    """
    # ============================
    # Metadata
    # ============================
    config = configuration["DATA_LOADER"]
    print("data loader config is \n", config)

    meta_data_list: list[pd.DataFrame] = []
    if config["smn-precip"]:
        meta_data_list.append(
            pd.read_csv(
                f"{config['data_folder']}ogd-smn-precip_meta_stations.csv",
                sep=";",
                encoding="windows-1252",
            )
        )
    if config["smn"]:
        meta_data_list.append(
            pd.read_csv(
                f"{config['data_folder']}ogd-smn_meta_stations.csv",
                sep=";",
                encoding="windows-1252",
            )
        )
    meta_data = pd.concat(meta_data_list).reset_index(drop=True)

    mask = np.ones(len(meta_data), dtype=bool)

    if config["lat_range"] is not None:
        mask = (
            mask
            & (meta_data.station_coordinates_wgs84_lat >= config["lat_range"][0])
            & (meta_data.station_coordinates_wgs84_lat < config["lat_range"][1])
        )
    if config["lon_range"] is not None:
        mask = (
            mask
            & (meta_data.station_coordinates_wgs84_lon >= config["lon_range"][0])
            & (meta_data.station_coordinates_wgs84_lon < config["lon_range"][1])
        )

    stations = meta_data.loc[mask, "station_abbr"].tolist()

    if len(config["stations"]) >= 1:
        stations = config["stations"]
    # ============================
    # Load the data
    # ============================
    # 1. read the dataset file architecture
    dataset = ds.dataset(
        f"{config['data_folder']}partitioned_parquets/",
        partitioning="hive",
    )  #  hive very important here since we partiontioned the dataset

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
    df["reference_timestamp"] = pd.to_datetime(df.reference_timestamp, format=FMT)
    return df, meta_data
