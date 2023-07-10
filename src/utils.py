import random
import shutil

from pathlib import Path

import torch
import numpy as np


def make_dir(path):
    path = Path(path)

    if path.is_dir():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)

    return path


def make_deterministic(seed=0, benchmark=True):
    seed = int(seed)

    if seed == -1:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = benchmark


def get_device(device_name):
    if device_name != 'cpu' and torch.cuda.is_available():
        device = torch.device(device_name)
    else:
        device = torch.device('cpu')

    return device
