from pathlib import Path

import yaml


def load_config(path=None) -> dict:
    config_path = Path(path) if path else Path(__file__).with_name("config.yaml")
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    cfg["image_size"] = tuple(cfg["image_size"])
    return cfg
