from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def select_device(requested_device):
    if requested_device != "auto":
        return torch.device(requested_device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
