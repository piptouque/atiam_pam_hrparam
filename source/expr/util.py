import numpy as np
import numpy.typing as npt


def find_win_start(x: npt.NDArray[float], attack_length: int) -> int:
    """Find the best region of interest for the windowing the signal

    Args:
        sig (npt.NDArray[float]): A temporal signal
        anticipation_length (int): Length to start from the max of the signal

    Returns:
        Tuple[int, int]: [description]
    """
    max_idx = np.argmax(x)
    idx_start = max(0, int(np.floor(max_idx - attack_length)))
    return idx_start


def make_linramp(size: int, start: int, stop: int) -> npt.NDArray[float]:
    """Linear force ramp

    Args:
        size (int): [description]
        start (int): [description]
        stop (int): [description]

    Returns:
        npt.NDArray[float]: [description]
    """
    r = np.zeros(size)
    r[start:stop] = np.arange(stop - start) / (stop - start)
    return r
