from collections import defaultdict
from pathlib import Path

import pandas as pd
from pandas import DataFrame

columns = list(
    pd.read_csv(
        "/home/lea/Documents/Data/meteoswiss-weather-stations/raw_files/abo/ogd-smn_abo_h_historical_2020-2029.csv.csv",
        sep=";",
        encoding="windows-1252",
    ).columns
)

# group files by (network, station_abbr)
station_files = defaultdict(list)
for csv_file in Path(
    "/home/lea/Documents/Data/meteoswiss-weather-stations/raw_files/"
).rglob("*_h_*.csv"):
    df_peek = pd.read_csv(csv_file, sep=";", encoding="windows-1252", nrows=1)
    network = "smn" if len(df_peek.columns) >= 4 else "smn-precip"
    station_abbr = df_peek["station_abbr"].values[0]
    station_files[(network, station_abbr)].append(csv_file)

# concatenate all periods per station and write one parquet
for (network, station_abbr), files in station_files.items():
    dfs = []
    for csv_file in sorted(files):
        df: DataFrame = pd.read_csv(csv_file, sep=";", encoding="windows-1252")
        for col in columns:
            if col not in df.columns:
                df[col] = float("nan")
        dfs.append(df)

    combined = (
        pd.concat(dfs).drop_duplicates(subset=["reference_timestamp"]).sort_values("reference_timestamp")
    )

    out = (
        Path(
            "/home/lea/Documents/Data/meteoswiss-weather-stations/partitioned_parquets/"
        )
        / f"network={network}"
        / f"{station_abbr}.parquet"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out)

