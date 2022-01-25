
import pandas as pd
import numpy as np
import numpy.typing as npt

import json
from types import SimpleNamespace


def load_data_json(path: str, cls=SimpleNamespace, **kwargs) -> object:
    """Constructs an object with `cls` factory method from json data at `path`.

    Args:
        path (str): Path to the json file,
        **kwargs: any additional arguments to be given to the constructor.

    Returns:
        object: constructed object.
    """
    with open(path, mode='r') as config:
        data = json.load(
            config, object_hook=lambda d: cls(**d, **kwargs))
        return data


def load_data_csv(path: str, cls=SimpleNamespace, **kwargs) -> object:
    """Constructs an object with `cls` factory method from csv data at `path`.

    Args:
        path (str): Path to the json file,
        **kwargs: any additional arguments to be given to the constructor.

    Returns:
        object: constructed object.
    """
    df = pd.read_csv(path)
    data_dict = df.to_dict()
    for (k, v) in df.items():
        data_dict[k] = v.to_numpy()
    data = cls(**data_dict, **kwargs)
    return data
