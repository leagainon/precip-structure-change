from pathlib import Path

import pandas as pd
from pandas import DataFrame

df_list: list[DataFrame] = []
columns = list(
    pd.read_csv(
        "/home/lea/Documents/Data/meteoswiss-weather-stations/abo/ogd-smn_abo_h_historical_2020-2029.csv.csv",
        sep=";",
        encoding="windows-1252",
    ).columns
)

for csv_file in Path(
    "/home/lea/Documents/Data/meteoswiss-weather-stations/raw_files/"
).rglob("*_h_*.csv"):
    df: DataFrame = pd.read_csv(csv_file, sep=";", encoding="windows-1252")

    df["network"] = "NA"
    if len(df.columns) > 2:
        df["network"] = "smn"
    else:
        df["network"] = "smn-precip"
        for col in columns:
            if not col in df.columns:
                df[col] = pd.NA
    out = (
        Path(
            "/home/lea/Documents/Data/meteoswiss-weather-stations/partitioned_parquets"
        )
        / f"{df['station_abbr'].values[0]}.parquet"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
