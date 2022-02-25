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
        nb_bands_bank: int = 1,
        decimation_factor_bank: int = None,
        order_filter_bank: int = 128,
        w_trans_bank: float = 0.05,
        smoothing_factor_noise: float = 1,
        quantile_ratio_noise: float = 1 / 3,
        ar_order_noise: int = 10,
        thresh_ratio_ester: float = 0.1,
        correct_alphas_noise: bool = True,
        discard_freq_esm: bool = False,
        clip_damp_esm: bool = False,
    ) -> None:

        self.fs = fs
        self.n_fft_noise = n_fft_noise
        #
        self.n_esprit = n_esprit
        #
        self.smoothing_factor_noise = smoothing_factor_noise
        self.quantile_ratio_noise = quantile_ratio_noise
        self.ar_order_noise = ar_order_noise
        self.correct_alphas_noise = correct_alphas_noise
        #
        self.p_max_ester = p_max_ester
        self.thresh_ratio_ester = thresh_ratio_ester
        #
        self.clip_damp_esm = clip_damp_esm
        self.discard_freq_esm = discard_freq_esm
        #
        #
        self.bank = None
        if decimation_factor_bank is None:
            decimation_factor_bank = nb_bands_bank // 2
        nb_bands_bank = int(nb_bands_bank)
        if nb_bands_bank > 1:
            self.bank = FilterBank(
                nb_bands_bank,
                decimation_factor_bank,
                order_filter=order_filter_bank,
                w_trans=w_trans_bank,
            )

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
        # FIX: works only on first band
        nb_bands_used = 1
        if self.bank is not None:
            x_bands = self.bank.process(x_white)
            zs_list = []
            alphas_list = []
            x_esm_list = []
            for b in range(nb_bands_used):
                x_band = x_bands[b]
                # 2. Estimate the ESM model order on the whitened signal
                # because ESTER does work quite as well with the whitened signal
                r_band = Ester.estimate_esm_order(
                    x_band, n=self.n_esprit, p_max=self.p_max_ester
                )
                # 3. Apply ESPRIT on the whitened signal
                # and estimate the ESM model parameters.
                w_cap, _ = Esprit.subspace_weighting_mats(x_band, self.n_esprit, r_band)
                phi_cap = Esprit.spectral_matrix(w_cap)
                zs, _ = Esprit.estimate_poles(
                    phi_cap,
                    clip_damp=self.clip_damp_esm,
                    discard_freq=self.discard_freq_esm,
                )
                # Discard frequencies that are outside the bandwidth of this band,
                # because they are already in the next/previous band's bandwidth.
                gammas_band, nus_band = EsmModel.poles_to_dampfreq(zs)
                nu_cutoff = 0.5 * self.bank.w_band * self.bank.decimation_factor
                ids_good = np.logical_and(-nu_cutoff <= nus_band, nus_band <= nu_cutoff)
                if not np.any(ids_good):
                    # Nothing here, skipping this band after all.
                    x_esm_list.append(None)
                    continue
                gammas_band = gammas_band[ids_good]
                nus_band = nus_band[ids_good]
                zs = EsmModel.dampfreq_to_poles(gammas_band, nus_band)
                # Compute the alphas before shifting back the frequencies!
                alphas = Esprit.estimate_esm_alphas(x_band, zs)
                # Shift back the normalised frequencies.
                nus_band_unshifted = (
                    nus_band / self.bank.decimation_factor + self.bank.nus_centre[b]
                )
                zs = EsmModel.dampfreq_to_poles(gammas_band, nus_band_unshifted)
                x_esm = EsmModel.from_complex(zs, alphas)
                #
                zs_list += zs.tolist()
                alphas_list += alphas.tolist()
                x_esm_list.append(x_esm)

            zs = np.asarray(zs_list)
            alphas = np.asarray(alphas_list)
            x_esm_bands = np.asarray(x_esm_list)
            x_esm = EsmModel.from_complex(zs, alphas)

        else:
            # 2. Estimate the ESM model order on the whitened signal
            # because ESTER does work quite as well with the whitened signal
            r = Ester.estimate_esm_order(
                x_white,
                self.n_esprit,
                self.p_max_ester,
                thresh_ratio=self.thresh_ratio_ester,
            )
            # 3. Apply ESPRIT on the whitened signal
            # and estimate the ESM model parameters.
            x_esm = Esprit.estimate_esm(
                x_white,
                self.n_esprit,
                r,
                clip_damp=self.clip_damp_esm,
                discard_freq=self.discard_freq_esm,
            )

        if self.correct_alphas_noise:
            # 4. Correct the real amplitudes and phases
            # from the estimated noise psd
            alphas_corr = NoiseWhitening.correct_alphas(
                x_esm.alphas, x_esm.zs, noise_psd
            )
            x_esm = EsmModel.from_complex(x_esm.zs, alphas_corr)

        # 5. estimate the noise part of the signal
        # TODO: Update the esm list with the corrected alphas..
        if self.bank is not None:
            x_noise_bands = np.zeros_like(x_bands)
            for b in range(nb_bands_used):
                x_band = x_bands[b]
                x_esm_band = x_esm_bands[b]
                if x_esm_band is not None:
                    x_noise_bands[b] = Esprit.estimate_noise(
                        x_band, self.n_esprit, x_esm_band.r
                    )
            x_noise = np.sum(x_noise_bands, axis=0)
        else:
            x_noise = Esprit.estimate_noise(x, self.n_esprit, r)
        return x_esm, x_noise, x_white
