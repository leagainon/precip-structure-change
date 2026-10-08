import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from .constants import FMT


def load_station_data(
    data_folder: str,
    lat_range: list[float] | None = None,
    lon_range: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    A data loader for meteoswiss stations data and metadata
    Inputs:
    ---
    - data_folder: string, path to the data data folder
    - lat_range : list of two floats, the range of latitude values in the interval -90° to 90°. Default is None (no filtering)
    - lon_range : list of two floats, the range of longitude values in the interval -180° to 90°. Default is None (no filtering)
    Outputs:
    ---
    - df : pandas dataframe containing the stations data properly formatted
    - metadata : pandas dataframe of the stations metadata
    """
    # ============================
    # Metadata
    # ============================
    meta_data_precip: pd.DataFrame = pd.read_csv(
        f"{data_folder}gd-smn-precip_meta_stations.csv",
        sep=";",
        encoding="windows-1252",
    )
    meta_data: pd.DataFrame = pd.read_csv(
        f"{data_folder}ogd-smn_meta_stations.csv",
        sep=";",
        encoding="windows-1252",
    )
    meta_data = pd.concat([meta_data_precip, meta_data]).reset_index(drop=True)

    mask = np.ones(len(meta_data), dtype=bool)

    if lat_range is not None:
        mask = (
            mask
            & (meta_data.station_coordinates_wgs84_lat >= lat_range[0])
            & (meta_data.station_coordinates_wgs84_lat < lat_range[1])
        )
    if lon_range is not None:
        mask = (
            mask
            & (meta_data.station_coordinates_wgs84_lon >= lon_range[0])
            & (meta_data.station_coordinates_wgs84_lon < lon_range[1])
        )

    stations = meta_data.loc[mask, "station_abbr"].tolist()

    # ============================
    # Load the data
    # ============================
    # 1. read the dataset file architecture
    dataset = ds.dataset(
        f"{data_folder}partitioned_parquets/",
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
