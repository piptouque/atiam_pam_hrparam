from typing import Tuple

import numpy as np
import numpy.typing as npt

from hr.esm import EsmModel
from hr.preprocess import NoiseWhitening, FilterBank
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
        ar_order_noise: int = 10,
        thresh_ratio_ester: float = 0.1,
        correct_alphas: bool = True,
        discard_freq: bool = False,
        clip_damp: bool = False,
    ) -> None:

        self.fs = fs
        self.n_fft_noise = n_fft_noise
        #
        self.n_esprit = n_esprit
        #
        self.smoothing_factor_noise = smoothing_factor_noise
        self.quantile_ratio_noise = quantile_ratio_noise
        self.ar_order_noise = ar_order_noise
        #
        self.p_max_ester = p_max_ester
        self.thresh_ratio_ester = thresh_ratio_ester
        #
        self.clip_damp = clip_damp
        self.discard_freq = discard_freq
        #
        self.correct_alphas = correct_alphas

    def perform(
        self, x: npt.NDArray[complex]
    ) -> Tuple[
        EsmModel, npt.NDArray[complex], npt.NDArray[complex], npt.NDArray[complex]
    ]:
        """Decomposes the termporal signal with an ESM model

        Args:
            x (npt.NDArray[complex]): [description]

        Returns:
            Tuple[ EsmModel, npt.NDArray[complex], npt.NDArray[complex], npt.NDArray[complex] ]: [description]
        """

        # 1. Whiten the noise
        x_white, noise_psd = NoiseWhitening.whiten(
            x,
            self.n_fft_noise,
            fs=self.fs,
            ar_order=self.ar_order_noise,
            quantile_ratio=self.quantile_ratio_noise,
            smoothing_factor=self.smoothing_factor_noise,
        )

        # 2. Estimate the ESM model order on the whitened signal
        # because ESTER does work quite as well with the whitened signal
        r = Ester.estimate_esm_order(
            x_white,
            self.n_esprit,
            self.p_max_ester,
            thresh_ratio=self.thresh_ratio_ester,
        )

        # If the signal is of a real data type,
        # The spectrum is symmetric so search for twice as many sines!
        if np.isrealobj(x):
            r *= 2

        # 3. Apply ESPRIT on the whitened signal
        # and estimate the ESM model parameters
        # The order to search for is twice the estimated,
        x_esm = Esprit.estimate_esm(
            x_white,
            self.n_esprit,
            r,
            clip_damp=self.clip_damp,
            discard_freq=self.discard_freq,
        )

        if self.correct_alphas:
            # 4. Correct the real amplitudes and phases
            # from the estimated noise psd
            alphas_corr = NoiseWhitening.correct_alphas(
                x_esm.alphas, x_esm.zs, noise_psd
            )
            x_esm = EsmModel.from_complex(x_esm.zs, alphas_corr)

        # TODO
        x_noise = Esprit.estimate_noise(x, self.n_esprit, r)

        return x_esm, x_noise, x_white
