import scipy.linalg as alg
import scipy.signal as sig
import scipy.ndimage as img
import numpy as np
import numpy.typing as npt

from typing import Tuple, List, Union

from hr.esm import EsmModel


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
        x_h = alg.hankel(x[:n], r=x[n - 1 :])
        l = x.shape[-1] - n + 1
        return x_h @ x_h.transpose().conj() / l

    @classmethod
    def subspace_weighting_mats(
        cls, x: npt.NDArray[complex], n: int, k: int
    ) -> Tuple[npt.NDArray[complex], npt.NDArray[complex]]:
        """Gets W and W_per the subspace weighting matrices used in ESPRIT algorithm

        Args:
            x (npt.NDArray[complex]): [description]
            n (int): [description]
            k (int): [description]

        Returns:
            Tuple[npt.NDArray[complex], npt.NDArray[float]]: (n,k) and (n,n-k) W and W_per matrices
        """
        r_xx = cls._correlation_mat(x, n)
        u_1, _, _ = alg.svd(r_xx)
        w = u_1[:, :k]
        w_per = u_1[:, k:]
        return w, w_per

    @classmethod
    def estimate_dampfreq(
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
        phi = alg.pinv(w_down) @ w_up
        zs = alg.eig(phi, left=False, right=False)
        log_zs = np.log(zs)
        # no damping should ever be negative.
        # fix: just discard those that are.
        gammas = -np.minimum(0, np.real(log_zs))  # damping factors
        nus = np.imag(log_zs) / (2 * np.pi)  # frequencies
        return gammas, nus

    @classmethod
    def estimate_amp(
        cls, x: npt.NDArray[float], gammas: npt.NDArray[float], nus: npt.NDArray[float]
    ) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
        """Estimate the amplitude in the ESM model using least-squares.

        Args:
            x (npt.NDArray[float]): Input signal
            gammas (npt.NDArray[float]): Array of estimated normalised damping factors
            nus (npt.NDArray[float]): Array of estimated normalised frequencies

        Returns:
            Tuple[npt.NDArray[float], npt.NDArray[float]]: (estimated real amplitudes, estimated initial phases)
        """
        n_s = len(x)  # signal's length
        ts = np.arange(n_s)  # array of discrete times
        z_logs = -gammas + 2j * np.pi * nus  # log of the pole
        v = np.exp(np.outer(ts, z_logs))  # Vandermonde matrix of dimension N
        alphas = alg.pinv(v) @ x
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
        w, w_per = cls.subspace_weighting_mats(x, n, k)
        gammas, nus = cls.estimate_dampfreq(w)
        amps, phis = cls.estimate_amp(x, gammas, nus)

        esm = EsmModel(gammas=gammas, nus=nus, amps=amps, phis=phis)

        return (esm, w, w_per)


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


class AdaptiveEsprit:
    """Centralised methods used in the Adaptive ESPRIT framework."""

    @staticmethod
    def _step_spectral_matrix(
        psi_prev: npt.NDArray[complex],
        e: npt.NDArray[complex],
        g: npt.NDArray[complex],
        w: npt.NDArray[complex],
        w_prev: npt.NDArray[complex],
    ) -> Tuple[npt.NDArray[complex], npt.NDArray[complex]]:
        """Sing
        See Badeau et al., 2005
        and HrHatrac David et al., 2006
        For reference. Same notations as the first article.

        Args:
            psi_prev (npt.NDArray[complex]): Previous value of the matrix $\Psi$
            e (npt.NDArray[complex]): Current value of the vector $e$, obtained with a stubspace tracking method
            g (npt.NDArray[complex]): Current value of the vector $g$, obtained with a subspace tracking method
            w (npt.NDArray[complex]): Current value of the subspace weighting matrix $W$, obtained with a subspace tracking method
            w_prev (npt.NDArray[complex]): Previous value of the subspace weighting matrix $W$, obtained with a subspace tracking method

        Returns:
            Tuple[npt.NDArray[complex], npt.NDArray[complex]]: ($\Phi$, $\Psi$)
        """
        # Init values of interest from parameters
        nu = w[-1].T.conj()
        nu_norm_sq = np.sum(np.abs(nu) ** 2)
        w_down_prev, w_up_prev = w_prev[:-1], w_prev[1:]
        e_down, e_up = e[:-1], e[1:]
        # Algorithm given in Table 1
        e_minus = w_down_prev.T.conj() @ e_up
        e_plus = w_up_prev.T.conj() @ e_down
        e_plus_ap = e_plus + g @ (e_up.T.conj() @ e_down)
        psi = psi_prev + e_minus @ g.T.conj() + g @ e_plus_ap.T.conj()
        phi_vec = psi.T.conj() @ nu
        phi_mat = psi + (nu @ phi_vec.T.conj()) / (1 - nu_norm_sq)
        return phi_mat, psi

    @staticmethod
    def _step_spectral_matrix_fae(
        psi_prev: npt.NDArray[complex],
        e: npt.NDArray[complex],
        g: npt.NDArray[complex],
        w: npt.NDArray[complex],
        w_prev: npt.NDArray[complex],
        phi_mat_prev: npt.NDArray[complex],
    ) -> Tuple[
        npt.NDArray[complex],
        npt.NDArray[complex],
        npt.NDArray[complex],
        npt.NDArray[complex],
    ]:
        """
        See Badeau et al., 2005, part 3. 'Spectral matrix tracking'
        For reference. Same notations as the first article.

        Args:
            psi_prev (npt.NDArray[complex]): Previous value of the matrix $\Psi$
            e (npt.NDArray[complex]): Current value of the vector $e$, obtained with a stubspace tracking method
            g (npt.NDArray[complex]): Current value of the vector $g$, obtained with a subspace tracking method
            w (npt.NDArray[complex]): Current value of the subspace weighting matrix $W$, obtained with a subspace tracking method
            w_prev (npt.NDArray[complex]): Previous value of the subspace weighting matrix $W$, obtained with a subspace tracking method

        Returns:
            Tuple[npt.NDArray[complex], npt.NDArray[complex], npt.NDArray[complex], npt.NDArray[complex]]: ($\Phi$, $\Psi$, $\bar{a}$, $\bar{b}$)
        """
        # Init values of interest from parameters
        nu = w[-1].T.conj()
        nu_norm_sq = np.sum(np.abs(nu) ** 2)
        w_down_prev, w_up_prev = w_prev[:-1], w_prev[1:]
        e_down, e_up = e[:-1], e[1:]
        # Algorithm given in Table 1
        e_minus = w_down_prev.T.conj() @ e_up
        e_plus = w_up_prev.T.conj() @ e_down
        e_plus_ap = e_plus + g @ (e_up.T.conj() @ e_down)
        psi = psi_prev + e_minus @ g.T.conj() + g @ e_plus_ap.T.conj()
        phi_vec = psi.T.conj() @ nu
        # phi_mat = psi + (nu @ phi_vec.T.conj()) / (1 - nu_norm_sq)
        # Additional stuff
        nu_prev = w_prev[-1].T.conj()
        nu_prev_norm_sq = np.sum(np.abs(nu_prev) ** 2)
        e_n = e[-1]
        #
        phi_vec_prev = psi_prev.conj().H @ nu_prev
        e_plus_apap = e_plus_ap + (e_n / (1 - nu_norm_sq)) * phi_vec
        delta_phi_vec = phi_vec / (1 - nu_norm_sq) - phi_vec_prev / (
            1 - nu_prev_norm_sq
        )
        #
        a_bar = np.stack((g, e_minus, nu_prev), axis=0)
        b_bar = np.stack((e_plus_apap, g, delta_phi_vec), axis=0)
        phi_mat = phi_mat_prev + a_bar @ b_bar.T.conj()
        return phi_mat, psi, a_bar, b_bar

    @staticmethod
    def _step_eigenvals_fae(
        g_mat_prev: npt.NDArray[complex],
        g_ap_mat_prev: npt.NDArray[complex],
        d_vec_prev: npt.NDArray[complex],
        phi_mat: npt.NDArray[complex],
        a_bar: npt.NDArray[complex],
        b_bar: npt.NDArray[complex],
    ) -> Tuple[npt.NDArray[complex], npt.NDArray[complex]]:
        """
        See Badeau et al., 2005, part 4. 'Eigenvalues tracking'
        For reference. Same notations as the first article.

        Args:
            g_mat_prev (npt.NDArray[complex]): [description]
            g_ap_mat_prev (npt.NDArray[complex]): [description]
            d_vec_prev (npt.NDArray[complex]): Previous eigenvalues as a vector
            a_bar (npt.NDArray[complex]): [description]
            b_bar (npt.NDArray[complex]): [description]

        Returns:
            Tuple[npt.NDArray[complex], npt.NDArray[complex]]: (Current eigenvalues vector $d$,Current right eigenvectors matrix $G$)
        """
        #
        a_bar_tilde = g_ap_mat_prev.T.conj() @ a_bar
        b_bar_tilde = g_mat_prev.T.conj() @ b_bar
        #
        def phi_bar(z: complex) -> npt.NDArray[complex]:
            car = 1 / (z - d_vec_prev)
            dar = b_bar_tilde.T.conj() @ np.diag(car) @ a_bar_tilde
            # TODO: what is I_bar?? I'm putting the identity for now.
            return np.identity(like=dar) - dar

        #
        g_tilde_mat = None  # TODO
        g_ap_mat = None  # TODO
        g_ap_tilde_mat = None  # TODO
        #
        g_mat = g_mat_prev @ g_tilde_mat
        g_ap_mat = g_ap_mat_prev @ g_ap_tilde_mat
        #
        d_mat = g_ap_mat.T.conj() @ phi_mat @ g_mat
        return d_mat, g_mat


class Fapi:
    """See Badeau et al., 2005"""

    @staticmethod
    def _step_spectral_weights_fapi(
        x: npt.NDArray[float],
        w_cap_prev: npt.NDArray[complex],
        z_cap_prev: npt.NDArray[float],
        beta: float = 0.99,
    ) -> Tuple[
        npt.NDArray[complex],
        npt.NDArray[float],
        npt.NDArray[complex],
        npt.NDArray[complex],
    ]:
        """[summary]

        Args:
            x (npt.NDArray[float]): [description]
            w_prev (npt.NDArray[complex]): [description]
            z_prev (npt.NDArray[float]): [description]
            beta (float, optional): [description]. Defaults to 0.99.

        Returns:
            Tuple[npt.NDArray[complex], npt.NDArray[float], npt.NDArray[complex], npt.NDArray[complex]]: [description]
        """
        y = w_cap_prev.T.conj() @ x
        h = z_cap_prev @ y
        g = h / (beta + y.T.conj() @ h)
        # $\epsilon^2$ in the article
        diff_sq = np.sum(np.abs(x) ** 2 - np.abs(y) ** 2)
        g_norm_sq = np.sum(np.abs(g) ** 2)
        tau = diff_sq / (1 + diff_sq * g + np.sqrt(1 + diff_sq * g_norm_sq))
        eta = 1 - tau * g_norm_sq
        y_ap = eta * y + tau * g
        h_ap = z_cap_prev.T.conj() @ y_ap
        epsilon_vec = (tau / eta) * (z_cap_prev @ g - (h_ap.T.conj() @ g) * g)
        z = (1 / beta) * (z_cap_prev - g @ h_ap.T.conj() + epsilon_vec @ g.T.conj())
        e_ap = eta * x - w_cap_prev @ y_ap
        w = w_cap_prev + e_ap @ g.T.conj()
        return w, z, e_ap, g

    @staticmethod
    def _step_spectral_weights_sw_fapi(
        x: npt.NDArray[float],
        w_cap_prev: npt.NDArray[complex],
        z_cap_prev: npt.NDArray[float],
        x_cap_prev: npt.NDArray[float],
        v_cap_hat_prev: npt.NDArray[float],
        beta: float = 0.99,
    ) -> Tuple[
        npt.NDArray[complex],
        npt.NDArray[float],
        npt.NDArray[complex],
        npt.NDArray[float],
        npt.NDArray[complex],
        npt.NDArray[complex],
    ]:
        """[summary]

        Args:
            x_vec (npt.NDArray[float]): [description]
            w_mat_prev (npt.NDArray[complex]): [description]
            z_mat_prev (npt.NDArray[float]): [description]
            x_mat_prev (npt.NDArray[float]): [description]
            v_mat_hat_prev (npt.NDArray[float]): [description]
            beta (float, optional): Forgetting factor. Defaults to 0.99.

        Returns:
            Tuple[ npt.NDArray[complex], npt.NDArray[float], npt.NDArray[complex], npt.NDArray[float], npt.NDArray[complex], npt.NDArray[complex] ]:
                ($W$, $Z$, $X$, $\hat{V}$, $e$, $g$)
        """

        # TODO: HOW TO COMPUTE INVERSE SQUARE ROOT??
        # this?: https://stackoverflow.com/a/66160829
        # don't think so..
        # some additional definitions
        l = x_cap_prev.shape[-1]
        r = w_cap_prev.shape[-1]
        # Rank of the update involved in equation (4)
        # p = 2 in the truncated window case.
        p = 2
        j_cap_bar = np.asarray([1, 0], [0, -(beta**l)])
        # Section similar to SW - PAST
        # Update the $x(t)$ vector
        x_prev_l = x_cap_prev[:, 0]
        # n x p matrix
        x_bar = np.stack((x, x_prev_l), axis=1)
        # and the $X(t)$ matrix
        x_cap = np.roll(x_cap_prev, shift=-1, axis=1)
        x_cap[:, -1] = x
        # Update the $y(t)$ and $\hat{v}(t-l)$ vectors
        # n x p matrix
        y = w_cap_prev.T.conj() @ x
        v_hat_prev_l = v_cap_hat_prev[:, 0]
        # and the $\hat{Y}(t)$ matrix
        y_cap_hat = np.roll(v_cap_hat_prev, shift=-1, axis=1)
        y_cap_hat[:, -1] = y
        v_prev_l = w_cap_prev.T.conj() @ x_prev_l
        #
        # a n x p matrix actually
        y_bar_hat = np.stack((y, v_hat_prev_l), axis=1)
        y_bar = np.stack((y, v_prev_l), axis=1)
        # an r x p matrix
        h_bar = z_cap_prev @ y_bar_hat
        # an r x p matrix
        g_bar = h_bar @ alg.inv(beta * alg.inv(j_cap_bar) + y_bar.T.conj() @ h)
        # TW - API main section
        epsilon_var_bar = alg.sqrtm(x_bar.T.conj() @ x_bar - y_bar.T.conj() @ y_bar)
        rho_bar = (
            np.identity(p)
            + epsilon_mat_bar.T.conj() @ (g_bar.T.conj() @ g_bar) @ epsilon_mat_bar
        )
        # p x p positive definite matrix
        tau_mat_bar = (
            epsilon_var_bar
            @ alg.inv(rho_bar + alg.sqrtm(rho_mat).T.conj())
            @ epsilon_var_bar.T.conj()
        )
        eta_mat_bar = np.identity(p) - (g_bar.T.conj() @ g_bar) @ tau_mat_bar
        y_bar_ap = y_bar @ eta_mat_bar + g_bar @ tau_mat_bar
        h_bar_ap = z_cap_prev.T.conj() @ y_bar_ap
        epsilon_bar = (z_cap_prev @ g_bar - g_bar @ (h_bar_ap.T.conj() @ g_bar)) @ (
            tau_mat_bar @ np.inv(eta_mat_bar)
        ).T.conj()
        z_mat = (1 / beta) * (
            z_cap_prev - g_bar @ h_bar_ap.T.conj() + epsilon_bar @ g_bar.T.conj()
        )
        # an n x p matrix
        e_mat_bar_ap = x_bar @ eta_bar - w_cap_prev @ y_bar_ap
        w_mat = w_cap_prev + e_mat_bar_ap @ g_bar.T.conj()
        v_mat_hat = y_mat - g_bar(g_bar @ tau_mat_bar).T.conj() @ y_mat
        return w_mat, z_mat, x_cap, v_mat_hat, e_mat_bar, g_bar


class Yast:
    """See Badeau et al., 2008"""

    @classmethod
    def step_spectral_weights(
        cls,
        x: npt.NDArray[float],
        w_cap_prev: npt.NDArray[complex],
        c_xx_prev: npt.NDArray[complex],
        c_yy_prev: npt.NDArray[complex],
        track_principal: bool,
        beta: float = 0.99,
        nb_it: int = 2,
        thresh: float = 0.01,
    ) -> Tuple[npt.NDArray[complex], npt.NDArray[complex], npt.NDArray[complex]]:
        """[summary]

        See Table 1, for the YAST algorithm

        Args:
            x (npt.NDArray[float]): [description]
            w_prev (npt.NDArray[complex]): [description]
            c_xx_prev (npt.NDArray[complex]): [description]
            c_yy_prev (npt.NDArray[complex]): [description]
            beta (float, optional): Forgetting factor. Defaults to 0.99.

        Returns:
            Tuple[npt.NDArray[complex], npt.NDArray[complex], npt.NDArray[complex]]: ($W(t)$, $C_{xx}(t)$, $C_{yy}(t)$)
        """
        y = w_cap_prev.T.conj() @ x
        e = x - w_cap_prev @ y
        sigma = alg.norm(e, ord=None)
        if np.isclose(sigma, 0):
            # no change from last time
            return w_cap_prev, c_xx_prev, c_yy_prev
        u = e / sigma
        x_ap = c_xx_prev @ x
        y_ap = c_yy_prev @ y
        y_apap = w_cap_prev.T.conj() @ x_ap
        # Added compared to Table 1
        c_xx = beta * c_xx_prev + x @ x.T.conj()
        #
        c_yy_ap = beta * c_yy_prev + y @ y.T.conj()
        z = beta * (y_apap - y_ap) / sigma + sigma * y
        gamma = sigma**2 + (beta / sigma**2) * (
            x.T.conj() @ x_ap - 2 * np.real(y.T.conj() @ y_apap) + y.T.conj() @ y_ap
        )
        # Filling $\bar{C}_{yy}(t)
        c_yy_bar = np.empty(c_yy_ap.shape + tuple(np.ones(c_yy_ap.ndim)))
        c_yy_bar[:-1, :-1] = c_yy_ap
        c_yy_bar[:-1, -1] = z.T.conj()
        c_yy_bar[-1, :-1] = z
        c_yy_bar[-1, -1] = gamma
        #
        phi_bar, lambd = cls._step_conjugate_gradient(
            c_yy_ap,
            c_yy_bar,
            z,
            track_principal=track_principal,
            nb_it=nb_it,
            thresh=thresh,
        )
        # Decomposition of vector phi_bar
        phi_small = np.abs(phi_bar[-1])
        assert phi_small <= 1, "Noooo"
        theta = phi_bar[-1] / phi_small
        epsilon = np.sqrt(1 - phi_small**2)
        phi = phi_bar[:-1] / (theta * epsilon)
        #
        e_1 = np.zeros_like(phi)
        e_1[0] = 1
        e_1 = -np.exp(1j * np.angle(phi[0])) * e_1
        #
        a = (phi - e_1) / alg.norm(phi - e_1, ord=None)
        b = w_cap_prev @ a
        q = w_cap_prev - 2 * b @ a.T.conj() - epsilon * u @ e_1.T.conj()
        q_1 = q[:, 0]
        d_vec = np.ones_like(phi)
        d_vec[0] = 1 / alg.norm(q_1, ord=None)
        d_mat = np.diag(d_vec)
        w = q @ d_mat
        #
        c_yy_ap_a = c_yy_ap @ a
        a_ap = 4 * (c_yy_ap_a - (a.T.conj() @ c_yy_ap_a) @ a)
        z_ap = 2 * z - 4 * (a.T.conj() @ z) @ a - epsilon * gamma * e_1
        #
        c_yy_apap = c_yy_ap - a_ap @ a.T.conj() - epsilon * z_ap @ e_1.T.conj()
        c_yy_apap = (c_yy_apap + c_yy_apap.T.conj()) / 2
        #
        c_yy = d_mat @ c_yy_apap @ d_mat
        return w, c_xx, c_yy

    @staticmethod
    def _step_conjugate_gradient(
        c_yy_ap: npt.NDArray[complex],
        c_yy_bar: npt.NDArray[complex],
        z: npt.NDArray[complex],
        track_principal: bool,
        nb_it: int = 1,
        thresh: float = 0.01,
    ) -> Tuple[npt.NDArray[complex], complex]:
        """See Table II, Conjugate Gradient algorithm

        Returns:
            [type]: [description]
        """
        if nb_it is None and thresh is None:
            thresh = 0.01
        g = z / alg.norm(z, ord=None)
        p = c_yy_ap @ g - (g.T.conj() @ c_yy_ap @ g) @ g
        #
        s = np.stack((np.zeros_like(g), p / alg.norm(p, ord=None), g), axis=1)
        s = np.concatenate((s, [1, 0, 0]), axis=0)
        #
        c_ys_bar = c_yy_bar @ s
        c_ss = s.T.conj() @ c_ys_bar
        print(c_ss.shape)
        assert c_ss.shape == (3, 3)
        k = 0
        delta_j = np.inf
        lambd = None
        theta_vec = None
        while (k is None or k < nb_it) and (
            thresh is None or alg.norm(delta_j) > thresh
        ):
            w, vr = alg.eig(c_ss, left=False, right=True)
            w_norm = arg.norm(w)
            idx_extr = np.argmax(w_norm) if track_principal else np.argmin(w_norm)
            theta_vec = w[idx_extr]
            lambd = vr[idx_extr]
            #
            theta_round = np.asarray(
                [
                    -alg.norm([theta_vec[1], theta_vec[2]], ord=2),
                    theta_vec[0].conj()
                    * (theta_vec[1] / np.abs(theta_vec[1]))
                    / np.sqrt(1 + np.abs(theta_vec[2] / theta_vec[1]) ** 2),
                    theta_vec[0].conj()
                    * (theta_vec[2] / np.abs(theta_vec[2]))
                    / np.sqrt(1 + np.abs(theta_vec[1] / theta_vec[2]) ** 2),
                ]
            )
            theta_mat = np.stack((theta_vec, theta_round), axis=1)
            s[:, :2] = s @ theta_mat
            c_ys_bar[:, :2] = c_ys_bar @ theta_mat
            c_ss[:2, :2] = theta_mat.T.conj() @ c_ss @ theta_mat
            delta_j = 2 * (c_ys_bar[:, 0] - lambd * s[:, 1])
            #
            g = delta_j / alg.norm(delta_j)
            g = g - s[:, :2] @ (s_[:, :2].T.conj() @ g)
            g = g / alg.norm(g)
            #
            s[:, 2] = g
            c_ys_bar[:, 2] = c_yy_bar @ s[:, 2]
            c_ss[:, 2] = s.T.conj() @ c_ys_var[:, 2]
            c_ss[2, :] = c_ss[:, 3].T.conj()
        phi_bar = s[:, 0]
        return phi_bar, lambd


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
        assert (
            1 <= p_max < n - 1
        ), f"Maximum ordre p_max={p_max} should be less than n-1={n-1}."

        w, _ = Esprit.subspace_weighting_mats(x, n, p_max)
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
            # 3. Compute e[p] from ksi_cap[p]
            mu_p = nu[p][-1]
            phi[p][:-1] = phi[p - 1] + mu_p * psi_l[p]
            phi[p][-1] = psi_r[p].T.conj() @ nu[p - 1] + mu_p * psi_lr[p].conj()
            w_cap_down_p = w_cap[p][:-1]
            e[p] = ksi_cap[p] - 1 / (1 - np.sum(np.abs(nu[p]) ** 2)) * np.outer(
                w_cap_down_p @ nu[p], phi[p].T.conj()
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
        j = np.asarray([1 / alg.norm(e[p], ord=None) ** 2 for p in range(len(e))])
        return j

    @classmethod
    def estimate_esm_ordre(
        cls, x: npt.NDArray[complex], n: int, p_max: int, thresh_ratio: float = 0.1
    ) -> int:
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
        j_max = np.amax(j)
        # first select peaks in signal
        j_max_thres_ids, _ = sig.find_peaks(j, height=j_max * thresh_ratio)
        # then filter peaks under threshold
        j_max_ids = sig.argrelextrema(j, np.greater_equal, order=1, mode="clip")[0]
        j_max_thres_ids = j_max_ids[np.nonzero(j[j_max_ids] >= j_max * thresh_ratio)[0]]
        # first index corresponds to p=1, second to p=2 etc.
        r = np.amax(j_max_thres_ids) + 1
        return r


class FiltreBank:
    """Some filtre bank design stuff"""

    def __init__(
        self, nb_bands: int, decimation_factor: int, scale: Union["lin", "log"]
    ) -> None:
        self.nb_bands = nb_bands
        self.scale = scale
        self.decimation_factor = decimation_factor

    def process(x: npt.NDArray[complex]) -> npt.NDArray[complex]:
        """[summary]

        Args:
            x (npt.NDArray[complex]): [description]

        Returns:
            npt.NDArray[complex]: (nb_bands, input_size // decimation factor)
        """
        assert x.ndim == 1
        x_bands = np.empty((self.nb_bands,) + x.shape)
