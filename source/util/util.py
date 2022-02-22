from types import SimpleNamespace
import json

import pandas as pd
import numpy as np
import numpy.typing as npt


def next_power_2(i: int) -> int:
    """Next power of two

    Args:
        i (int): _description_

    Returns:
        int: _description_
    """
    power = 1
    while power < i:
        power *= 2
    return power


def compute_frf(
    x: npt.NDArray[np.complex128], y: npt.NDArray[np.complex128]
) -> npt.NDArray[np.complex128]:
    """Computes the Frequency Response Function (FRF) of an (input, output) couple of spectres.

    Args:
        x (npt.NDArray[np.complex128]): Spectre of input signal
        y (npt.NDArray[np.complex128]): Spectre of output signal

    Returns:
        npt.NDArray[np.complex128]: FRF of output y with input x.
    """
    s_yx = y * np.conj(x)
    s_xx = x * np.conj(x)
    return s_yx / s_xx


def to_db(data: npt.NDArray[float]) -> npt.NDArray[float]:
    """[summary]

    Args:
        data (npt.NDArray[float]): [description]

    Returns:
        npt.NDArray[float]: [description]
    """
    return 10 * np.log10(data)


def snr(signal: npt.NDArray[float], noise: npt.NDArray[float]) -> float:
    """Computes signal to noise ratio (SNR) of a noisy signal and its noise.

    Args:
        signal (npt.NDArray[float]): [description]
        noise (npt.NDArray[float]): [description]

    Returns:
        float: [description]
    """
    return to_db(np.sum(signal**2) / np.sum(noise**2))


def load_data_json(path: str, cls=SimpleNamespace, **kwargs) -> object:
    """Constructs an object with `cls` factory method from json data at `path`.

    Args:
        path (str): Path to the json file,
        **kwargs: any additional arguments to be given to the constructor.

    Returns:
        object: constructed object.
    """
    with open(path, mode="r") as config:
        data = json.load(config, object_hook=lambda d: cls(**d, **kwargs))
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
