from typing import Union

import scipy.linalg as alg
import scipy.signal as sig
import scipy.ndimage as img
import numpy as np
import numpy.typing as npt

from util.util import next_power_2


class PreEmphasis:
    @staticmethod
    def emphasise(sig: npt.NDArray[float], b=np.array([1, -0.95])):
        """
        Pre-emphasize the signal with an FIR of order 2.

        Parameters
        ----------
        `sig`: the input signal
        `b`: coefficients of the filter's transfer function

        Return
        ------
        `preemph_sig`: pre-emphasized signal
        """
        sig_emph = signal.lfilter(b, [1], sig)
        return sig_emph


class NoiseWhitening:
    """[summary]"""

    @staticmethod
    def _estimate_noise_psd(
        x_psd: npt.NDArray[complex], quantile_ratio: float, nu_width: float
    ) -> npt.NDArray[complex]:
        """Estimates the noise's PSD with a median filter (smoothing the signal's PSD)
        Args:
            x_psd (npt.NDArray[complex]): [description]
            quantile_ratio (float): rank of the rank filter as a ratio of the width of the filter (in [0, 1]).
            nu_width (float): Width ratio for the rank filter, in normalised frequency (in [0, 1]).

        Returns:
            npt.NDArray[complex]: Estimated PSD of the noise.
        """
        n_fft = x_psd.shape[-1]
        size = int(np.round(nu_width * n_fft))
        # choose rank in function of the chosen quantile
        rank = int(np.round(quantile_ratio * size))
        noise_psd = img.rank_filter(x_psd, rank=rank, size=size)
        return noise_psd

    @classmethod
    def estimate_noise_psd(
        cls,
        x: npt.NDArray[complex],
        n_fft: int,
        fs: float = 1,
        quantile_ratio: float = 1 / 3,
        smoothing_factor: float = 1,
    ) -> npt.NDArray[complex]:
        """Estimates the noise's PSD from a temporal signal.

        Args:
            x (npt.NDArray[complex]): [description]
            n_fft (int): Number of frequency bins
            fs (float): Sampling rate
            quantile_ratio (float): rank of the rank filter as a ratio of the width of the filter (in [0, 1]).
            smoothing_factor (float): Width ratio for the median filter used in noise PSD estimation, in window width.

        Returns:
            npt.NDArray[complex]: [description]
        """
        # x: the input signal FOR EACH FREQUENCY BAND
        # smoothing_order: at least two times the length of the PSD's principal lobe (can be done visually)
        # AR_order: ~ 10
        # note: no need to give it the correct sample frequency,
        # since it is only used to return the fftfreqs.
        win_name = "hann"
        m = n_fft // 2
        win_width_norm = 2  # for a Hann window
        win_width = win_width_norm / m
        # At least twice the bandwidth of the window used in the PSD computation!
        nu_width = smoothing_factor * 2 * win_width
        # print(f"width={nu_width * fs} Hz, {nu_width * n_fft} samples")
        _, x_psd = sig.welch(
            x, fs=fs, nfft=n_fft, window=win_name, nperseg=m, return_onesided=False
        )

        # Step 2: estimating the noise's PSD with a median filter (smoothing the signal's PSD)
        noise_psd = cls._estimate_noise_psd(
            x_psd, quantile_ratio=quantile_ratio, nu_width=nu_width
        )
        return noise_psd

    @classmethod
    def estimate_noise_ar_coeffs(
        cls,
        x: npt.NDArray[complex],
        n_fft: int,
        fs: float = 1,
        ar_order: int = 10,
        quantile_ratio: float = 1 / 3,
        smoothing_factor: float = 1,
    ) -> npt.NDArray[complex]:
        """Estimate the coefficients of the filter generating the coloured noise from a white one.

        Args:
            x (npt.NDArray[complex]): Input temporal signal, for each frequency band
            n_fft (int): Number of frequential bins
            fs (float): Sampling rate
            ar_order (int): order of the estimated AR filter. ~ 10?
            quantile_ratio (float): rank of the rank filter as a ratio of the width of the filter (in [0, 1]).
            smoothing_factor (float): Width ratio for the median filter used in noise PSD estimation, in window width.
                At least 1 (=twice the length of the PSD's principal lobe (can be done visually))

        Returns:
            npt.NDArray[complex]: Coefficients of the AR filter.
        """
        noise_psd = cls.estimate_noise_psd(
            x,
            n_fft=n_fft,
            fs=fs,
            quantile_ratio=quantile_ratio,
            smoothing_factor=smoothing_factor,
        )
        # Step 3: calculating the autocovariance of the noise
        # autocovariance (vector) of the noise
        ac_coeffs = np.real(np.fft.ifft(noise_psd, n=n_fft))
        # coefficients matrix of the Yule-Walker system
        # = autocovariance matrix with the last row and last column removed
        r_mat = alg.toeplitz(ac_coeffs[: ar_order - 1], ac_coeffs[: ar_order - 1])
        # the constant column of the Yule-Walker system
        r = ac_coeffs[1:ar_order].T
        # the AR coefficients (indices 1, ..., N-1)
        b = -np.asarray(alg.pinv(r_mat, return_rank=False) @ r)
        # the AR coefficients (indices 0, ..., N-1)
        b = np.insert(b, 0, 1)
        return b

    @classmethod
    def whiten(
        cls,
        x: npt.NDArray[complex],
        n_fft: int,
        fs: float = 1,
        quantile_ratio: float = 1 / 3,
        ar_order: int = 10,
        smoothing_factor: float = 1,
    ) -> npt.NDArray[complex]:
        """Whiten the noise in input temporal signal

        Args:
            x (npt.NDArray[complex]): Input temporal signal
            n_fft (int): Number of frequential bins
            fs (float): Sampling rate
            ar_order (int): order of the estimated AR filter
            quantile_ratio (float): rank of the rank filter as a ratio of the width of the filter (in [0, 1]).
            smoothing_factor (float): Width ratio for the median filter used in noise PSD estimation, in window width.

        Returns:
            npt.NDArray[complex]: Temporal signal with noise whitened.
        """
        b = cls.estimate_noise_ar_coeffs(
            x,
            n_fft=n_fft,
            fs=fs,
            ar_order=ar_order,
            quantile_ratio=quantile_ratio,
            smoothing_factor=smoothing_factor,
        )
        # Step 4: applying the corresponding FIR to the signal's PSD to obtain the whitened signal
        # The FIR is the inverse of the AR filter so the coefficients of the FIR's numerator
        # are the coefficients of the AR's denominator, i.e. the array b
        # denominator coeff of the FIR's transfer function is 1
        x_whitened = sig.lfilter(b, [1], x)
        return x_whitened


class FilterBank:
    """Some filter bank design stuff"""

    def __init__(
        self,
        nb_bands: int,
        decimation_factor: int,
        order_filter: int,
        w_trans: float,
    ) -> None:
        """_summary_

        Args:
            nb_bands (int): _description_
            order_filter (int, optional): Number of taps in each filter = order of filter + 1 (choose it odd for the hp filter to function well).
            w_trans (int, optional): Transition width (in normalised frequency).
        """
        assert nb_bands >= 2
        # In order to avoid aliasing, decimate by less than the number of bands.
        assert (
            decimation_factor <= nb_bands
        ), "Decimation factor should be less than the number of bands."
        # self.nb_bands = nb_bands
        self.decimation_factor = decimation_factor
        self.nus_centre = 0.5 * np.arange(nb_bands) / (nb_bands - 1)

        self.filter_coeffs = self.make_filters(
            nb_bands, order_filter=order_filter, w_trans=w_trans
        )

    @property
    def nb_bands(self) -> int:
        return np.shape(self.filter_coeffs)[0]

    def process(self, x: npt.NDArray[complex]) -> npt.NDArray[complex]:
        """[summary]

        Args:
            x (npt.NDArray[complex]): [description]

        Returns:
            npt.NDArray[complex]: (nb_bands, input_size // nb_bands)
        """
        assert x.ndim == 1
        n_cap = len(x)
        x_filtered = np.asarray(
            [sig.lfilter(coeffs, [1], x) for coeffs in self.filter_coeffs]
        )
        x_mods = np.exp(2j * np.pi * np.outer(self.nus_centre, np.arange(n_cap)))
        x_shifted = x_filtered * x_mods
        x_bands = sig.decimate(x_shifted, self.decimation_factor, axis=-1)

        return x_bands

    def make_filters(
        self, nb_bands: int, order_filter: int, w_trans: float
    ) -> npt.NDArray[float]:
        """Create a filter bank with Remez' algorithm.
        Parameters ----------
            nb_bands (int)
            order_filter (int): Number of taps in each filter = order of filter + 1 (choose it odd for the hp filter to function well).
            w_trans (float): Transition width (in normalised frequency).
        Return
        ------
        `coeffs`: an array containing the numpy arrays of the channel's coefficients
        """
        w_band = self.nus_centre[1] - self.nus_centre[0]
        assert (
            w_band > w_trans
        ), f"Bandwidth={w_band} should be superior to width of transistory={w_trans}"

        # calculating band edges
        band_lp = [
            0,
            w_band / 2,
            w_band / 2 + w_trans / 2,
            0.5,
        ]  # first band for a lowpass filter
        amp_lp = [1, 0]  # corresponding desired amplitudes

        coeffs_lp = sig.remez(order_filter, band_lp, amp_lp)

        coeffs = [coeffs_lp]

        for i in range(1, nb_bands - 1):
            band_bp = [
                0,
                self.nus_centre[i] - w_band / 2 - w_trans / 2,
                self.nus_centre[i] - w_band / 2,
                self.nus_centre[i] + w_band / 2,
                self.nus_centre[i] + w_band / 2 + w_trans / 2,
                0.5,
            ]  # ith band for a bandpass filter
            amp_bp = [0, 1, 0]  # corresponding desired amplitudes

            coeffs_bp = sig.remez(order_filter, band_bp, amp_bp)

            coeffs.append(coeffs_bp)

        band_hp = [
            0,
            0.5 - w_band / 2 - w_trans / 2,
            0.5 - w_band / 2,
            0.5,
        ]  # last band for a highpass filter
        amp_hp = [0, 1]  # corresponding desired amplitudes

        coeffs_hp = sig.remez(order_filter, band_hp, amp_hp)

        coeffs.append(coeffs_hp)

        return coeffs
