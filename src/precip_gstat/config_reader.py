import tomllib


def read_config(config_fp):
    with open(config_fp, mode="rb") as fp:
        config = tomllib.load(fp)
    print("config is \n", config)
    return config
