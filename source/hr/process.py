import scipy.linalg
import scipy.signal as sig
import numpy as np
import numpy.typing as npt

from typing import Tuple, List

from hr.util import EsmModel


class SubspaceDecomposition:
    """[summary]"""

    def __init__(self, x: npt.NDArray[complex], sr: float) -> None:
        self.x = x
        self.sr = sr

        self.x_white = None
        self.w = None
        self.w_per = None
        self.esm = None

    def setup(self, n: int, p_max: int, n_fft: int) -> None:
        """[summary]

        Args:
            n (int): [description]
            p_max (int): [description]
            n_fft (int): [description]
        """

        # 1. Whiten the noise because ESTER does not
        # seem to work quite as well with coloured noise
        ar_ordre = 10
        self.x_white = NoiseWhitening.whiten(
            self.x, n_fft, fs=self.sr, ar_ordre=ar_ordre
        )

        # 2. Estimate the ESM model ordre on the whitened signal
        thresh_ratio = 0.1
        r = Ester.estimate_esm_ordre(self.x_white, n, p_max, thresh_ratio=thresh_ratio)

        # 3. Apply ESPRIT on the whitened signal in ordre to estimate
        esm, w, w_per = Esprit.estimate_esm(self.x_white, n, r)
        self.esm = esm
        self.w = w
        self.w_per = w_per


class Esprit:
    """[summary]"""

    @staticmethod
    def _correlation_mat(x: npt.NDArray[complex], n: int) -> npt.NDArray[complex]:
        """Gets correlation matrix used in ESPRIT algorithm

        Args:
            x (npt.NDArray[complex]): [description]
            n (int): [description]

        Returns:
            npt.NDArray[complex]: [description]
        """
        x_h = scipy.linalg.hankel(x[:n], r=x[n - 1 :])
        l = x.shape[-1] - n + 1
        return x_h @ x_h.transpose().conj() / l

    @classmethod
    def spectral_mats(
        cls, x: npt.NDArray[complex], n: int, k: int
    ) -> Tuple[npt.NDArray[complex], npt.NDArray[complex]]:
        """Gets W and W_per used in ESPRIT algorithm

        Args:
            x (npt.NDArray[complex]): [description]
            n (int): [description]
            k (int): [description]

        Returns:
            Tuple[npt.NDArray[complex], npt.NDArray[float]]: (n,k) and (n,n-k) W and W_per matrices
        """
        r_xx = cls._correlation_mat(x, n)
        u_1, _, _ = scipy.linalg.svd(r_xx)
        # sigma = scipy.linalg.diagsvd(s, r_xx.shape[0], r_xx.shape[1])
        w = u_1[:, :k]
        w_per = u_1[:, k:]
        return w, w_per

    @classmethod
    def _estimate_dampfreq(
        cls,
        w: npt.NDArray[complex],
    ) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
        """Estimates the normalised frequencies and normalised damping factors
        for the ESM model using the ESPRIT algorithm.

        Args:
            x (np.ndarray): input signal
            n (int): number of lines in the Hankel matrix S
            k (int): number of searched sinusoids

        Returns:
            Tuple[npt.NDArray[float], npt.NDArray[float]]: (estimated normalised frequencies, estimated normalised dampings)
        """
        w_down = w[:-1]
        w_up = w[1:]
        phi = scipy.linalg.pinv(w_down) @ w_up
        zs = scipy.linalg.eig(phi, left=False, right=False)
        deltas = -np.log(np.abs(zs))  # damping factors
        nus = np.angle(zs) / (2 * np.pi)  # frequencies
        return deltas, nus

    @classmethod
    def _estimate_amp(
        cls, x: npt.NDArray[float], deltas: npt.NDArray[float], nus: npt.NDArray[float]
    ) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
        """Estimate the amplitude in the ESM model using least-squares.

        Args:
            x (npt.NDArray[float]): Input signal
            deltas (npt.NDArray[float]): Array of estimated normalised damping factors
            nus (npt.NDArray[float]): Array of estimated normalised frequencies

        Returns:
            Tuple[npt.NDArray[float], npt.NDArray[float]]: (estimated real amplitudes, estimated initial phases)
        """
        n_s = len(x)  # signal's length
        ts = np.arange(n_s)  # array of discrete times
        z_logs = -deltas + 2j * np.pi * nus  # log of the pole
        v = np.exp(np.outer(ts, z_logs))  # Vandermonde matrix of dimension N
        alphas = scipy.linalg.pinv(v) @ x
        amps = np.abs(alphas)
        phis = np.angle(alphas)
        return amps, phis

    @classmethod
    def estimate_esm(
        cls, x: npt.NDArray[float], n: int, k: int
    ) -> Tuple[EsmModel, npt.NDArray[complex], npt.NDArray[complex]]:
        """Estimates the complete ESM model using the ESPRIT algorithm.

        Args:
            x (np.ndarray): input signal
            n (int): number of lines in the Hankel matrix S
            k (int): number of searched sinusoids

        Returns:
            Tuple[EsmModel, npt.NDArray[complex], npt.NDArray[complex]]: (EsmModel, Signal spectral matrix, Noise spectral matrix)
        """
        w, w_per = cls.spectral_mats(x, n, k)
        deltas, nus = cls._estimate_dampfreq(w)
        amps, phis = cls._estimate_amp(x, deltas, nus)

        esm = EsmModel(deltas=deltas, nus=nus, amps=amps, phis=phis)

        return (esm, w, w_per)


class NoiseWhitening:
    """[summary]"""

    @staticmethod
    def _estimate_noise_psd(
        x_psd: npt.NDArray[complex], nu_width: float
    ) -> npt.NDArray[complex]:
        """Estimates the noise's PSD with a median filter (smoothing the signal's PSD)
        Args:
            x_psd (npt.NDArray[complex]): [description]
            smoothing_factor (float): Width ratio for the median filtre used in noise PSD estimation, in window width.

        Returns:
            npt.NDArray[complex]: Estimated PSD of the noise.
        """
        n_fft = x_psd.shape[-1]
        size = int(np.round(nu_width * n_fft))

        noise_psd = scipy.ndimage.median_filter(x_psd, size=size)
        return noise_psd

    @classmethod
    def estimate_noise_psd(
        cls,
        x: npt.NDArray[complex],
        n_fft: int,
        fs: float = 1,
        smoothing_factor: float = 1,
    ) -> npt.NDArray[complex]:
        """Estimates the noise's PSD from a temporal signal.

        Args:
            x (npt.NDArray[complex]): [description]
            n_fft (int): Number of frequency bins
            smoothing_factor (float): Width ratio for the median filtre used in noise PSD estimation, in window width.
            fs (float): Sampling rate

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
        _, x_psd = sig.welch(x, fs=fs, nfft=n_fft, window=win_name, nperseg=m)

        # Step 2: estimating the noise's PSD with a median filtre (smoothing the signal's PSD)
        noise_psd = cls._estimate_noise_psd(x_psd, nu_width)
        return noise_psd

    @classmethod
    def estimate_noise_ar_coeffs(
        cls,
        x: npt.NDArray[complex],
        n_fft: int,
        fs: float = 1,
        ar_ordre: int = 10,
        smoothing_factor: float = 1,
    ) -> npt.NDArray[complex]:
        """Estimate the coefficients of the filtre generating the coloured noise from a white one.

        Args:
            x (npt.NDArray[complex]): Input temporal signal, for each frequency band
            n_fft (int): Number of frequential bins
            ar_ordre (int): Ordre of the estimated AR filtre. ~ 10?
            smoothing_factor (float): Width ratio for the median filtre used in noise PSD estimation, in window width.
                At least 1 (=twice the length of the PSD's principal lobe (can be done visually))
            fs (float): Sampling rate

        Returns:
            npt.NDArray[complex]: Coefficients of the AR filtre.
        """
        noise_psd = cls.estimate_noise_psd(
            x, n_fft=n_fft, fs=fs, smoothing_factor=smoothing_factor
        )
        # Step 3: calculating the autocovariance of the noise
        # autocovariance (vector) of the noise
        ac_coeffs = np.real(np.fft.ifft(noise_psd, n=n_fft))
        # coefficients matrix of the Yule-Walker system
        # = autocovariance matrix with the last row and last column removed
        r_mat = scipy.linalg.toeplitz(
            ac_coeffs[: ar_ordre - 1], ac_coeffs[: ar_ordre - 1]
        )
        # the constant column of the Yule-Walker system
        r = ac_coeffs[1:ar_ordre].T
        # the AR coefficients (indices 1, ..., N-1)
        b = -scipy.linalg.pinv(r_mat) @ r
        # the AR coefficients (indices 0, ..., N-1)
        b = np.insert(b, 0, 1)
        return b

    @classmethod
    def whiten(
        cls,
        x: npt.NDArray[complex],
        n_fft: int,
        fs: float = 1,
        ar_ordre: int = 10,
        smoothing_factor: float = 1,
    ) -> npt.NDArray[complex]:
        """Whiten the noise in input temporal signal

        Args:
            x (npt.NDArray[complex]): Input temporal signal
            n_fft (int): Number of frequential bins
            ar_ordre (int): Ordre of the estimated AR filtre
            smoothing_factor (float): Width ratio for the median filtre used in noise PSD estimation, in window width.
            fs (float): Sampling rate

        Returns:
            npt.NDArray[complex]: Temporal signal with noise whitened.
        """
        b = cls.estimate_noise_ar_coeffs(
            x, n_fft=n_fft, fs=fs, ar_ordre=ar_ordre, smoothing_factor=smoothing_factor
        )
        # Step 4: applying the corresponding FIR to the signal's PSD to obtain the whitened signal
        # The FIR is the inverse of the AR filter so the coefficients of the FIR's numerator
        # are the coefficients of the AR's denominator, i.e. the array b
        # denominator coeff of the FIR's transfer function is 1
        x_whitened = sig.lfilter(b, [1], x)
        return x_whitened


class Ester:
    """See Badeau et al., 2006"""

    @staticmethod
    def error(
        x: npt.NDArray[complex], n: int, p_max: int
    ) -> List[npt.NDArray[complex]]:
        """[summary]

        Still don't know if p_max whether chosen or estimated,
        and how.
        Don't discard the bogus p=0 case,
        because then the best ordre is the argmax!

        Args:
            x (npt.NDArray[complex]): [description]
            n (int): [description]
            p_max (int): Maximum ordre to consider, in [|1, n-2|]

        Returns:
            List[npt.NDArray[complex]]: Estimation error of p for p in [1, p_max]
        """
        # x: input signal
        # n: n: number of lines in the Hankel matrix S
        # K: ?
        assert (
            1 <= p_max < n - 1
        ), f"Maximum ordre p_max={p_max} should be less than n-1={n-1}."

        w, _ = Esprit.spectral_mats(x, n, p_max)
        w_cap = [w[:, :p] for p in range(p_max + 1)]
        #
        # nus are (p)-vecs
        nu = [w_cap[p][-1].T.conj() for p in range(p_max + 1)]
        #
        psi_l = [None] * (p_max + 1)
        psi_r = [None] * (p_max + 1)
        psi_lr = [None] * (p_max + 1)
        ksi_cap = [np.zeros((n - 1, p), dtype=w_cap[0].dtype) for p in range(p_max + 1)]
        phi = [np.zeros(p, dtype=w_cap[0].dtype) for p in range(p_max + 1)]
        # norm(e[p]) is always in [0, 1], see original paper
        e = [np.empty((p, p), dtype=w_cap[0].dtype) for p in range(p_max + 1)]
        for p in range(1, p_max + 1):
            # Recursive computation of e[p]
            # see Badeau et al. (2006) Table 1 for details on the algorithm
            w_cap_down_pm = w_cap[p - 1][:-1]
            w_cap_up_pm = w_cap[p - 1][1:]
            w_down_p = w_cap[p][:-1, p - 1]
            w_up_p = w_cap[p][1:, p - 1]
            # 1. Update of the auxiliary matrix psi_mat[p]
            psi_r[p] = w_cap_down_pm.T.conj() @ w_up_p
            psi_l[p] = w_cap_up_pm.T.conj() @ w_down_p
            psi_lr[p] = w_down_p.T.conj() @ w_up_p
            # 2. Update of the auxiliary matrix ksi_mat[p]
            ksi_p = w_up_p - w_cap_down_pm @ psi_r[p] - w_down_p * psi_lr[p]
            psi_l_p = psi_l[p]
            ksi_cap[p][:, :-1] = ksi_cap[p - 1] - np.outer(w_down_p, psi_l_p.conj())
            ksi_cap[p][:, -1] = ksi_p
            # 3. Computation of e[p] from ksi_cap[p]
            mu_p = nu[p][-1]
            phi[p][:-1] = phi[p - 1] + mu_p * psi_l[p]
            phi[p][-1] = psi_r[p].T.conj() @ nu[p - 1] + mu_p * psi_lr[p].conj()
            w_cap_down_p = w_cap[p][:-1]
            e[p] = ksi_cap[p] - 1 / (1 - np.linalg.norm(nu[p], ord=2) ** 2) * np.outer(
                (w_cap_down_p @ nu[p]), phi[p].T.conj()
            )
        # discard p=0 case (meaningless)
        e = e[1:]
        return e

    @classmethod
    def inverse_error_func(
        cls, x: npt.NDArray[complex], n: int, p_max: int
    ) -> npt.NDArray[float]:
        """Gets the inverse error function J in the ESTER algorithm

        Args:
            x (npt.NDArray[complex]): [description]
            n (int): [description]
            p_max (int): [description]

        Returns:
            npt.NDArray[float]: [description]
        """
        e = cls.error(x, n, p_max)
        j = np.array([1 / np.linalg.norm(e[p], ord=2) ** 2 for p in range(len(e))])
        return j

    @classmethod
    def estimate_esm_ordre(
        cls, x: npt.NDArray[complex], n: int, p_max: int, thresh_ratio: float = 0.1
    ) -> npt.NDArray[float]:
        """Gets the estimated ESM model ordre r using the ESTER algorithm.
        see: http://ieeexplore.ieee.org/document/1576975/

        'In presence of noise, [...] a robust way of selecting
        the modeling order consists in detecting the greatest value
        of p for which the function J reaches a local maximum which
        is greater than a fraction of its global maximum
        (typically one tenth of the global maximum)'

        Args:
            x (npt.NDArray[complex]): [description]
            n (int): [description]
            p_max (int): [description]
            thresh_ratio (float): fraction of the global maximum of J used as a threshold

        Returns:
            npt.NDArray[float]: [description]
        """
        j = cls.inverse_error_func(x, n, p_max)
        j_max = np.max(j)
        # first select peaks in signal
        j_max_thres_ids, _ = sig.find_peaks(j, height=j_max * thresh_ratio)
        # then filter peaks under threshold
        j_max_ids = sig.argrelextrema(j, np.greater_equal, order=1, mode="clip")[0]
        j_max_thres_ids = j_max_ids[np.nonzero(j[j_max_ids] >= j_max * thresh_ratio)[0]]
        # first index corresponds to p=1, second to p=2 etc.
        r = np.max(j_max_thres_ids) + 1
        return r
