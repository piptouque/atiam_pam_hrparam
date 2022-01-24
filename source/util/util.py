
import pandas as pd
import numpy as np
import numpy.typing as npt

import json


def load_data_json(cls, path: str, **kwargs) -> object:
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


def load_data_csv(cls, path: str, **kwargs) -> object:
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


def make_modetime_dataframe(data: npt.NDArray[float], n: npt.NDArray[int], t: npt.NDArray[float]) -> pd.DataFrame:
    """Construct a pandas DataFrame from a 2-d numpy vector.
    see: https: // moonbooks.org/Articles/How-to-store-a-multidimensional-matrix-in-a-dataframe-with-pandas-/

    Args:
        data(npt.NDArray[float]): 2-d numpy vector data
        n(npt.NDArray[int]): 0-axis
        t(npt.NDArray[float]): 1-axis

    Returns:
        pd.DataFrame:
    """
    df_indices = pd.MultiIndex.from_product([n, t], names=["n", "t"])
    data = np.repeat(data, len(t), axis=0)
    df = pd.DataFrame(data=data, index=df_indices, columns=t)
    return df
