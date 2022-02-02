
import numpy as np
import numpy.typing as npt

from typing import Tuple


def find_win_extends(data: npt.NDArray[float], win_length: int) -> Tuple[int, int]:
    """Find the best region of interest
    for a windowing the signal

    Args:
        data (npt.NDArray[float]): A temporal signal
        win_length (int): The desired length of the window.

    Returns:
        Tuple[int, int]: Start and end indices for the window
    """
    max_idx = np.argmax(data)
    idx_start = int(np.floor(max_idx - (win_length-1)/2))
    idx_stop = idx_start + win_length
    return idx_start, idx_stop
