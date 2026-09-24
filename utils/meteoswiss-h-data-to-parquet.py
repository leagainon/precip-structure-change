from pathlib import Path

import pandas as pd
from pandas import DataFrame

df_list: list[DataFrame] = []
for csv_file in Path("/home/lea/Documents/Data/meteoswiss-weather-stations/").rglob(
    "*_h_*.csv"
):
    df: DataFrame = pd.read_csv(csv_file, sep=";", encoding="windows-1252")
    df_list.append(df)

merged: DataFrame = (
    pd.concat(df_list).set_index(["reference_timestamp", "station_abbr"]).sort_index()
)
merged.to_parquet("precip_hourly.parquet")
