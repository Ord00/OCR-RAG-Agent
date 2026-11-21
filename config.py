from loaden import load_config

config = load_config("config.yaml")

CACHE_HOST = config["cache"]["host"]
CACHE_PORT = int(config["cache"]["port"])
DB_HOST = config["database"]["host"]
DB_PORT = int(config["database"]["port"])