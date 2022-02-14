from typing import Tuple

import numpy.typing as npt

from hr.esm import EsmModel
from hr.preprocess import NoiseWhitening, FiltreBank
from hr.process import Esprit, Ester, Fapi, MiscAdaptiveTracking


class EsmSubspaceDecomposer:
    """[summary]"""

    def __init__(
        self,
        fs: float,
        n_esprit: int,
        p_max_ester: int,
        n_fft_noise: int,
        smoothing_factor_noise: float = 1,
        quantile_ratio_noise: float = 1 / 3,
        ar_ordre_noise: int = 10,
        thresh_ratio_ester: float = 0.1,
    ) -> None:

        self.fs = fs
        self.n_fft_noise = n_fft_noise
        #
        self.n_esprit = n_esprit
        #
        self.smoothing_factor_noise = smoothing_factor_noise
        self.quantile_ratio_noise = quantile_ratio_noise
        self.ar_ordre_noise = ar_ordre_noise
        #
        self.p_max_ester = p_max_ester
        self.thresh_ratio_ester = thresh_ratio_ester

    def perform(
        self, x: npt.NDArray[complex]
    ) -> Tuple[
        EsmModel, npt.NDArray[complex], npt.NDArray[complex], npt.NDArray[complex]
    ]:
        """[summary]

        Args:
            n (int): [description]
            p_max (int): [description]
            n_fft (int): [description]
        """

        # 1. Whiten the noise
        x_white = NoiseWhitening.whiten(
            x,
            self.n_fft_noise,
            fs=self.fs,
            ar_ordre=self.ar_ordre_noise,
            quantile_ratio=self.quantile_ratio_noise,
            smoothing_factor=self.smoothing_factor_noise,
        )

        # 2. Estimate the ESM model ordre on the whitened signal
        # because ESTER does work quite as well with the whitened signal
        r = Ester.estimate_esm_ordre(
            x_white,
            self.n_esprit,
            self.p_max_ester,
            thresh_ratio=self.thresh_ratio_ester,
        )

        # 3. Apply ESPRIT on the whitened signal
        # and estimate the ESM model parameters
        x_esm, w, w_per = Esprit.estimate_esm(x_white, self.n_esprit, r)
        return x_esm, w, w_per, x_white
