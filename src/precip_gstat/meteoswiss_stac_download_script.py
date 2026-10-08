from pathlib import Path

import requests
from pystac_client import Client

output_dir = Path("/home/lea/Documents/Data/meteoswiss-weather-stations")
collection = "ch.meteoschweiz.ogd-smn"  # add -precip to download also the additional stations that have only gage measurements.

client = Client.open("https://data.geo.admin.ch/api/stac/v1")
collection = client.get_collection(collection)

list_assets = []
for item in collection.get_all_items():
    for asset_key, asset in item.assets.items():
        dest = output_dir / item.id / f"{asset_key}.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            print(f"file {dest} already downloaded")
            continue
        resp = requests.get(asset.href)
        with open(dest, "wb") as f:
            _ = f.write(resp.content)

        print(f"downloaded {dest.name}")
