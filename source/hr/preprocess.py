from typing import Union

import scipy.linalg as alg
import scipy.signal as sig
import scipy.ndimage as img
import numpy as np
import numpy.typing as npt


class NoiseWhitening:
    """[summary]"""

    @staticmethod
    def _estimate_noise_psd(
        x_psd: npt.NDArray[complex], quantile_ratio: float, nu_width: float
    ) -> npt.NDArray[complex]:
        """Estimates the noise's PSD with a median filter (smoothing the signal's PSD)
        Args:
            x_psd (npt.NDArray[complex]): [description]
            quantile_ratio (float): rank of the rank filtre as a ratio of the width of the filtre (in [0, 1]).
            nu_width (float): Width ratio for the rank filtre, in normalised frequency (in [0, 1]).

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
            quantile_ratio (float): rank of the rank filtre as a ratio of the width of the filtre (in [0, 1]).
            smoothing_factor (float): Width ratio for the median filtre used in noise PSD estimation, in window width.

        Returns:
            npt.NDArray[complex]: [description]
        """
        # x: the input signal FOR EACH FREQUENCY BAND
        # smoothing_ordre: at least two times the length of the PSD's principal lobe (can be done visually)
        # AR_ordre: ~ 10
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

        # Step 2: estimating the noise's PSD with a median filtre (smoothing the signal's PSD)
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
        ar_ordre: int = 10,
        quantile_ratio: float = 1 / 3,
        smoothing_factor: float = 1,
    ) -> npt.NDArray[complex]:
        """Estimate the coefficients of the filtre generating the coloured noise from a white one.

        Args:
            x (npt.NDArray[complex]): Input temporal signal, for each frequency band
            n_fft (int): Number of frequential bins
            fs (float): Sampling rate
            ar_ordre (int): Ordre of the estimated AR filtre. ~ 10?
            quantile_ratio (float): rank of the rank filtre as a ratio of the width of the filtre (in [0, 1]).
            smoothing_factor (float): Width ratio for the median filtre used in noise PSD estimation, in window width.
                At least 1 (=twice the length of the PSD's principal lobe (can be done visually))

        Returns:
            npt.NDArray[complex]: Coefficients of the AR filtre.
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
        r_mat = alg.toeplitz(ac_coeffs[: ar_ordre - 1], ac_coeffs[: ar_ordre - 1])
        # the constant column of the Yule-Walker system
        r = ac_coeffs[1:ar_ordre].T
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
        ar_ordre: int = 10,
        smoothing_factor: float = 1,
    ) -> npt.NDArray[complex]:
        """Whiten the noise in input temporal signal

        Args:
            x (npt.NDArray[complex]): Input temporal signal
            n_fft (int): Number of frequential bins
            fs (float): Sampling rate
            ar_ordre (int): Ordre of the estimated AR filtre
            quantile_ratio (float): rank of the rank filtre as a ratio of the width of the filtre (in [0, 1]).
            smoothing_factor (float): Width ratio for the median filtre used in noise PSD estimation, in window width.

        Returns:
            npt.NDArray[complex]: Temporal signal with noise whitened.
        """
        b = cls.estimate_noise_ar_coeffs(
            x,
            n_fft=n_fft,
            fs=fs,
            ar_ordre=ar_ordre,
            quantile_ratio=quantile_ratio,
            smoothing_factor=smoothing_factor,
        )
        # Step 4: applying the corresponding FIR to the signal's PSD to obtain the whitened signal
        # The FIR is the inverse of the AR filter so the coefficients of the FIR's numerator
        # are the coefficients of the AR's denominator, i.e. the array b
        # denominator coeff of the FIR's transfer function is 1
        x_whitened = sig.lfilter(b, [1], x)
        return x_whitened


class FiltreBank:
    """Some filtre bank design stuff"""

    def __init__(
        self, nb_bands: int, decimation_factor: int, scale: Union["lin", "log"]
    ) -> None:
        self.nb_bands = nb_bands
        self.scale = scale
        self.decimation_factor = decimation_factor

    def process(self, x: npt.NDArray[complex]) -> npt.NDArray[complex]:
        """[summary]

        Args:
            x (npt.NDArray[complex]): [description]

        Returns:
            npt.NDArray[complex]: (nb_bands, input_size // decimation factor)
        """
        assert x.ndim == 1
        x_bands = np.empty((self.nb_bands,) + x.shape)
