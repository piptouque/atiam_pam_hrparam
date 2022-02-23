from typing import Tuple, List, Union

import scipy.io.wavfile as wav
import scipy.linalg as alg
import scipy.signal as sig
import scipy.ndimage as img
import numpy as np
import tqdm

rng = np.random.default_rng(123)
import numpy.typing as npt


from hr.esm import EsmModel, BlockEsmModel
from hr.preprocess import NoiseWhitening, FilterBank


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
        assert np.ndim(x) == 1 and len(x) > n, "Insufficient number of samples."
        x_cap = alg.hankel(x[:n], r=x[n - 1 :])
        l = len(x) - n + 1
        return x_cap @ x_cap.T.conj() / l

    @classmethod
    def subspace_weighting_mats(
        cls, x: npt.NDArray[complex], n: int, r: int
    ) -> Tuple[npt.NDArray[complex], npt.NDArray[complex]]:
        """Gets W and W_per the subspace weighting matrices used in ESPRIT algorithm

        Args:
            x (npt.NDArray[complex]): [description]
            n (int): [description]
            r (int): [description]

        Returns:
            Tuple[npt.NDArray[complex], npt.NDArray[float]]: (n,r) and (n,n-r) W and W_per matrices
        """
        c_cap_xx = cls._correlation_mat(x, n)
        u_1, _, _ = alg.svd(c_cap_xx, compute_uv=True)
        w_cap = u_1[:, :r]
        w_cap_per = u_1[:, r:]
        return w_cap, w_cap_per

    @classmethod
    def spectral_matrix(
        cls,
        w_cap: npt.NDArray[complex],
    ) -> npt.NDArray[complex]:
        """[summary]
        Args:
            w_cap (npt.NDArray[complex]): Subspace weighting matrix
        Returns:
            npt.NDArray[complex]: $\Phi$
        """

        w_cap_down = w_cap[:-1]
        w_cap_up = w_cap[1:]
        phi_cap = alg.pinv(w_cap_down) @ w_cap_up
        return phi_cap

    @classmethod
    def partner_matrices(
        cls,
        w_cap: npt.NDArray[complex],
    ) -> Tuple[npt.NDArray[complex], npt.NDArray[complex]]:
        """Get the matrices $(\Psi, \Omega)$such that $\Phi = \Omega \Psi$
        Args:
            w_cap (npt.NDArray[complex]): Subspace weighting matrix
        Returns:
            npt.NDArray[complex]: ($\Psi$, $\Omega$)
        """

        w_cap_down = w_cap[:-1]
        w_cap_up = w_cap[1:]
        psi_cap = w_cap_down.T.conj() @ w_cap_up
        omega_cap = alg.pinv(w_cap_down.T.conj() @ w_cap_down)
        return psi_cap, omega_cap

    @classmethod
    def estimate_poles(
        cls,
        phi_cap: npt.NDArray[complex],
        clip_damp: bool = False,
        discard_freq: bool = False,
    ) -> Tuple[npt.NDArray[complex], npt.NDArray[complex]]:
        """Estimates the poles of the spectral matrix

        Args:
            phi_cap (npt.NDArray[complex]): _description_
            phi_cap (npt.NDArray[complex]): Spectral matrix
            clip_damp (bool, optional): Clips damping factors to [0, +inf[. Defaults to False.
            discard_freq (bool, optional): Discard poles whose estimated frequency is not in [-0.5, 0.5]. Defaults to False.

        Returns:
            npt.NDArray[complex], npt.NDArray[complex]]: (estimated poles, right eigenvectors)
        """
        zs, g_cap = alg.eig(phi_cap, left=False, right=True)
        zs = EsmModel.fix_poles(zs, clip_damp=clip_damp, discard_freq=discard_freq)
        return zs, g_cap

    @classmethod
    def estimate_esm_alphas(
        cls, x: npt.NDArray[float], zs: npt.NDArray[complex]
    ) -> Tuple[
        npt.NDArray[float], npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]
    ]:
        """Estimates the normalised frequencies and normalised damping factors,
        amplitude and initial phase
        for the ESM model using the ESPRIT algorithm.

        Args:
            phi_cap (npt.NDArray[complex]): Spectral matrix

        Returns:
            Tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]: (estimated normalised frequencies, estimated normalised dampings, estimated real amplitudes, estimated initial phases)
        """
        log_zs = np.log(zs)
        # array of discrete times
        ts = np.arange(len(x))
        # Vandermonde matrix of dimension N
        v_mat = np.exp(np.outer(ts, log_zs))
        alphas = np.dot(alg.pinv(v_mat), x)
        return alphas

    @classmethod
    def estimate_esm(
        cls,
        x: npt.NDArray[float],
        n: int,
        r: int,
        clip_damp: bool = False,
        discard_freq: bool = False,
    ) -> EsmModel:
        """Estimates the complete ESM model using the ESPRIT algorithm.
        Args:
            x (np.ndarray): input signal
            n (int): number of lines in the Hankel matrix S
            r (int): number of searched sinusoids

        Returns:
            Tuple[EsmModel, npt.NDArray[complex], npt.NDArray[complex]]: (EsmModel, Signal spectral matrix, Noise spectral matrix)
        """
        # n_cap = len(x)
        # TODO: set default for n according to the Cramer-Rao bounds!
        w_cap, _ = cls.subspace_weighting_mats(x, n, r)
        phi_cap = cls.spectral_matrix(w_cap)
        zs, _ = cls.estimate_poles(
            phi_cap, clip_damp=clip_damp, discard_freq=discard_freq
        )
        alphas = cls.estimate_esm_alphas(x, zs)
        #
        x_esm = EsmModel.from_complex(zs, alphas)
        return x_esm

    @classmethod
    def estimate_noise(
        cls, x: npt.NDArray[float], n: int, r: int
    ) -> npt.NDArray[complex]:
        """Estimate noise in input signal by projecting onto the noise subspace.
        Args:
            x (npt.NDArray[float]): _description_
            n (int): _description_
            r (int): _description_

        Returns:
            npt.NDArray[complex]: _description_
        """
        w_cap, _ = cls.subspace_weighting_mats(x, n, r)
        n = w_cap.shape[0]
        # projection matrix onto the harmonic signal
        p_cap_harm = w_cap @ w_cap.T.conj()
        x_cap = alg.hankel(x[:n], r=x[n - 1 :])
        x_cap_noise = (np.identity(n) - p_cap_harm) @ x_cap
        # take only the coefficients from 0 to n-1, then from n to n_cap
        # there are multiple possibilities, not sure what's best.
        x_noise = np.concatenate((x_cap_noise[:, 0], x_cap_noise[-1, 1:]))
        return x_noise


class BlockEsprit:
    """Esprit by block: non adaptive"""

    @classmethod
    def estimate_esm(
        cls, x: npt.NDArray[float], n: int, r: int, l_win: int, step: int
    ) -> BlockEsmModel:
        """Computes `esprit` and `least_square` by blocks.

        Arguments:
            sig (npt.NDArray[float]): the full-length input signal
            n (int): number of lines in the Hankel matrix S
            r (int): signal space dimension = number of sinusoids (n-K: noise space dimension)
            l_win (int): the window size (in samples)
            step (int): the hop size (in samples)
        Returns:
        """
        # the length of the signal (in samples)
        n_cap = len(x)
        nb_blocks = (n_cap - l_win) // step + 1
        x_esm_list = []
        for i in range(nb_blocks):
            idx_start = step * i
            idx_stop = step * i + l_win
            # ith truncated signal
            x_block = x[idx_start:idx_stop]

            x_esm = Esprit.estimate_esm(x, n, r)
            x_esm_list.append(x_esm)
        x_esm = BlockEsmModel(x_esm_list)
        return x_esm


class MiscAdaptiveTracking:
    """Some stuff from
    Badeau et al., 2005,
    and David et al., 2006
    and David et al., 2007
    """

    @staticmethod
    def track_spectral_matrix(
        e: npt.NDArray[complex],
        g: npt.NDArray[complex],
        w_cap: npt.NDArray[complex],
        w_cap_prev: npt.NDArray[complex],
        psi_cap_prev: npt.NDArray[complex],
    ) -> Tuple[npt.NDArray[complex], npt.NDArray[complex]]:
        """
        See Badeau et al., 2005
        For reference. Same notations as the first article.

        Args:
            e (npt.NDArray[complex]): Current value of the vector $e$, obtained with a stubspace tracking method
            g (npt.NDArray[complex]): Current value of the vector $g$, obtained with a subspace tracking method
            w (npt.NDArray[complex]): Current value of the subspace weighting matrix $W$, obtained with a subspace tracking method
            w_prev (npt.NDArray[complex]): Previous value of the subspace weighting matrix $W$, obtained with a subspace tracking method
            psi_prev (npt.NDArray[complex]): Previous value of the matrix $\Psi$

        Returns:
            Tuple[npt.NDArray[complex], npt.NDArray[complex]]: ($\Phi$, $\Psi$)
        """
        assert np.ndim(e) == 1 and np.ndim(g) == 1
        # Init values of interest from parameters
        nu = w_cap[-1].T.conj()
        nu_norm_sq = np.sum(np.abs(nu) ** 2)
        w_cap_down_prev, w_cap_up_prev = w_cap_prev[:-1], w_cap_prev[1:]
        e_down, e_up = e[:-1], e[1:]
        # Algorithm given in Table 1
        e_minus = w_cap_down_prev.T.conj() @ e_up
        e_plus = w_cap_up_prev.T.conj() @ e_down
        e_plus_ap = e_plus + g @ (e_up.T.conj() @ e_down)
        psi_cap = psi_cap_prev + e_minus @ g.T.conj() + g @ e_plus_ap.T.conj()
        phi = psi_cap.T.conj() @ nu
        phi_cap = psi_cap + (nu @ phi.T.conj()) / (1 - nu_norm_sq)
        return phi_cap, psi_cap

    @staticmethod
    def track_spectral_matrix_fae(
        e: npt.NDArray[complex],
        g: npt.NDArray[complex],
        w_cap: npt.NDArray[complex],
        w_cap_prev: npt.NDArray[complex],
        psi_cap_prev: npt.NDArray[complex],
        phi_cap_prev: npt.NDArray[complex],
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
            e (npt.NDArray[complex]): Current value of the vector $e$, obtained with a stubspace tracking method
            g (npt.NDArray[complex]): Current value of the vector $g$, obtained with a subspace tracking method
            w (npt.NDArray[complex]): Current value of the subspace weighting matrix $W$, obtained with a subspace tracking method
            w_prev (npt.NDArray[complex]): Previous value of the subspace weighting matrix $W$, obtained with a subspace tracking method
            psi_prev (npt.NDArray[complex]): Previous value of the matrix $\Psi$
            phi_cap_prev (npt.NDArray[complex]): Previous value of the matrix $\Phi$

        Returns:
            Tuple[npt.NDArray[complex], npt.NDArray[complex], npt.NDArray[complex], npt.NDArray[complex]]: ($\Phi$, $\Psi$, $\bar{a}$, $\bar{b}$)
        """
        assert np.ndim(e) == 1 and np.ndim(g) == 1
        # Init values of interest from parameters
        nu = w_cap[-1].T.conj()
        nu_norm_sq = np.sum(np.abs(nu) ** 2)
        # w_id_down = nu @ nu.T.conj()
        # w_id_down_err = alg.norm(nu)
        w_cap_down_prev, w_cap_up_prev = w_cap_prev[:-1], w_cap_prev[1:]
        e_down, e_up = e[:-1], e[1:]
        # Algorithm given in Table 1
        e_minus = w_cap_down_prev.T.conj() @ e_up
        e_plus = w_cap_up_prev.T.conj() @ e_down
        e_plus_ap = e_plus + g * np.dot(e_up.conj(), e_down)
        psi_cap = (
            psi_cap_prev + np.outer(e_minus, g.conj()) + np.outer(g, e_plus_ap.conj())
        )
        phi = psi_cap.T.conj() @ nu
        # Additional stuff
        nu_prev = w_cap_prev[-1].T.conj()
        nu_prev_norm_sq = np.sum(np.abs(nu_prev) ** 2)
        e_n = e[-1]
        #
        phi_prev = psi_cap_prev.conj().T.conj() @ nu_prev
        e_plus_apap = e_plus_ap + (e_n / (1 - nu_norm_sq)) * phi
        delta_phi = phi / (1 - nu_norm_sq) - phi_prev / (1 - nu_prev_norm_sq)
        #
        a_bar = np.stack((g, e_minus, nu_prev), axis=1)
        b_bar = np.stack((e_plus_apap, g, delta_phi), axis=1)
        phi_cap = phi_cap_prev + a_bar @ b_bar.T.conj()
        return phi_cap, psi_cap, a_bar, b_bar

    @staticmethod
    def track_poles_fae(
        phi_cap: npt.NDArray[complex],
        a_bar: npt.NDArray[complex],
        b_bar: npt.NDArray[complex],
        g_cap_prev: npt.NDArray[complex],
        g_cap_ap_prev: npt.NDArray[complex],
        d_prev: npt.NDArray[complex],
    ) -> Tuple[npt.NDArray[complex], npt.NDArray[complex]]:
        """
        See Badeau et al., 2005, part 4. 'Eigenvalues tracking'
        For reference. Same notations as the first article.

        Args:
            phi_cap (npt.NDArray[complex]): [description]
            a_bar (npt.NDArray[complex]): [description]
            b_bar (npt.NDArray[complex]): [description]
            g_cap_prev (npt.NDArray[complex]): [description]
            g_ap_cap_prev (npt.NDArray[complex]): [description]
            d_prev (npt.NDArray[complex]): Previous eigenvalues as a vector

        Returns:
            Tuple[npt.NDArray[complex], npt.NDArray[complex]]: (Current eigenvalues vector $d$,Current right eigenvectors matrix $G$)
        """
        """
        a_bar_tilde = g_cap_ap_prev.T.conj() @ a_bar
        b_bar_tilde = g_cap_prev.T.conj() @ b_bar
        #
        def phi_bar(z: complex) -> npt.NDArray[complex]:
            car = 1 / (z - d_prev)
            dar = b_bar_tilde.T.conj() @ np.diag(car) @ a_bar_tilde
            # TODO: what is I_bar?? I'm putting the identity for now.
            return np.identity(like=dar) - dar

        #
        g_cap_tilde = None  # TODO
        g_cap_ap = None  # TODO
        g_cap_tilde_ap = None  # TODO
        #
        g_cap = g_cap_prev @ g_cap_tilde
        g_cap_ap = g_cap_ap_prev @ g_cap_tilde_ap
        #
        d_cap = g_cap_ap.T.conj() @ phi_cap @ g_cap
        d = np.diagonal(d_cap)
        return d, g_cap
        """
        raise NotImplementedError()


class Hrhatrac:
    """and HrHatrac David et al., 2006
    For reference. Same notations as the article.
    """

    @staticmethod
    def track_poles(
        phi_cap: npt.NDArray[complex],
        d_prev: npt.NDArray[complex],
        g_cap_prev: npt.NDArray[complex],
        mu_d: float = 0.99,
        mu_g: float = 0.99,
    ) -> Tuple[npt.NDArray[complex], npt.NDArray[complex]]:
        """[summary]

        Args:
            phi_cap (npt.NDArray[complex]): [description]
            d_prev (npt.NDArray[complex]): Previous right eigenvectors
            g_cap_prev (npt.NDArray[complex]): Previous eigenvalues as a vector
            mu_d (float, optional): [description]. Defaults to 0.99.
            mu_g (float, optional): [description]. Defaults to 0.99.

        Returns:
            Tuple[npt.NDArray[complex], npt.NDArray[complex]]: (Current eigenvalues vector $d$,Current right eigenvectors matrix $G$)
        """
        assert 0 < mu_d < 1 and 0 < mu_g < 1
        #
        d = (1 - mu_d) * d_prev + mu_d * np.diagonal(
            alg.inv(g_cap_prev) @ phi_cap @ g_cap_prev
        )
        e_cap_g_prev = g_cap_prev - phi_cap @ g_cap_prev @ np.diag(1 / d)
        g_cap = (1 - mu_g) * g_cap_prev + mu_g * (
            phi_cap @ g_cap_prev @ np.diag(1 / d)
            + phi_cap.T.conj() @ e_cap_g_prev @ np.diag(1 / d.T.conj())
        )
        g_norm = alg.norm(g_cap, axis=0, keepdims=True)
        g_cap = g_cap / g_norm
        return d, g_cap


class Fapi:
    """See Badeau et al., 2005"""

    @staticmethod
    def track_spectral_weights(
        x: npt.NDArray[float],
        w_cap_prev: npt.NDArray[complex],
        z_cap_prev: npt.NDArray[complex],
        beta: float = 0.99,
    ) -> Tuple[
        npt.NDArray[complex],
        npt.NDArray[complex],
        npt.NDArray[complex],
        npt.NDArray[complex],
    ]:
        """[summary]

        Args:
            x (npt.NDArray[float]): (n)-dimensional vector
            w_prev (npt.NDArray[complex]): (n, r) matrix. Spectral weights.
            z_prev (npt.NDArray[complex]): [description]
            beta (float, optional): Forgetting factor. Defaults to 0.99.

        Returns:
            Tuple[npt.NDArray[complex], npt.NDArray[complex], npt.NDArray[complex], npt.NDArray[complex]]: [description]
        """
        assert 0 < beta < 1
        y = w_cap_prev.T.conj() @ x
        h = z_cap_prev @ y
        g = h / (beta + np.dot(y.conj(), h))
        g_norm_sq = np.sum(np.abs(g) ** 2)
        # $\epsilon_var^2$ in the article
        epsilon_var_sq = np.sum(np.abs(x) ** 2) - np.sum(np.abs(y) ** 2)
        tau = epsilon_var_sq / (
            1 + epsilon_var_sq * g_norm_sq + np.sqrt(1 + epsilon_var_sq * g_norm_sq)
        )
        eta = 1 - tau * g_norm_sq
        y_ap = eta * y + tau * g
        h_ap = z_cap_prev.T.conj() @ y_ap
        epsilon = (tau / eta) * (z_cap_prev @ g - np.dot(h_ap.conj(), g) * g)
        # print(h_ap[-3:])
        z_cap = (1 / beta) * (
            z_cap_prev - np.outer(g, h_ap.conj()) + np.outer(epsilon, g.conj())
        )
        e_ap = eta * x - w_cap_prev @ y_ap
        w_cap = w_cap_prev + np.outer(e_ap, g.conj())
        return w_cap, z_cap, e_ap, g

    @staticmethod
    def track_spectral_weights_tw(
        x: npt.NDArray[float],
        w_cap_prev: npt.NDArray[complex],
        z_cap_prev: npt.NDArray[complex],
        x_cap_prev: npt.NDArray[float],
        v_cap_hat_prev: npt.NDArray[complex],
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
            w_cap_prev (npt.NDArray[complex]): [description]
            z_cap_prev (npt.NDArray[complex]): [description]
            x_cap_prev (npt.NDArray[float]): Shape (n x l). Previous Hankel matrix.
            v_cap_hat_prev (npt.NDArray[complex]): [description]
            beta (float, optional): Forgetting factor. Defaults to 0.99.

        Returns:
            Tuple[ npt.NDArray[complex], npt.NDArray[float], npt.NDArray[complex], npt.NDArray[float], npt.NDArray[complex], npt.NDArray[complex] ]:
                ($W$, $Z$, $X$, $\hat{V}$, $e$, $g$)
        """
        # Rank of the update involved in equation (4)
        # p = 2 in the truncated window case.
        l = x_cap_prev.shape[-1]
        p = 2
        j_cap_bar = np.asarray([[1, 0], [0, -(beta**l)]])
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
        # and the $Y(t)$ matrix
        # (there's a mistake in Table II, it is indeed $Y(t)$ and not $\hat{Y}(t)$)
        y_cap = np.roll(v_cap_hat_prev, shift=-1, axis=1)
        y_cap[:, -1] = y
        v_prev_l = w_cap_prev.T.conj() @ x_prev_l
        #
        # a n x p matrix actually
        y_bar_hat = np.stack((y, v_hat_prev_l), axis=1)
        y_bar = np.stack((y, v_prev_l), axis=1)
        # an r x p matrix
        h_bar = z_cap_prev @ y_bar_hat
        # an r x p matrix
        g_bar = h_bar @ alg.inv(beta * alg.inv(j_cap_bar) + y_bar.T.conj() @ h_bar)
        # TW - API main section
        epsilon_var_bar = alg.sqrtm(x_bar.T.conj() @ x_bar - y_bar.T.conj() @ y_bar)
        rho_bar = (
            np.identity(p)
            + epsilon_var_bar.T.conj() @ (g_bar.T.conj() @ g_bar) @ epsilon_var_bar
        )
        # p x p positive definite matrix
        tau_bar = (
            epsilon_var_bar
            @ alg.inv(rho_bar + alg.sqrtm(rho_bar).T.conj())
            @ epsilon_var_bar.T.conj()
        )
        eta_bar = np.identity(p) - (g_bar.T.conj() @ g_bar) @ tau_bar
        y_bar_ap = y_bar @ eta_bar + g_bar @ tau_bar
        h_bar_ap = z_cap_prev.T.conj() @ y_bar_ap
        epsilon_bar = (z_cap_prev @ g_bar - g_bar @ (h_bar_ap.T.conj() @ g_bar)) @ (
            tau_bar @ alg.inv(eta_bar)
        ).T.conj()
        z_cap = (1 / beta) * (
            z_cap_prev - g_bar @ h_bar_ap.T.conj() + epsilon_bar @ g_bar.T.conj()
        )
        # an n x p matrix
        e_bar_ap = x_bar @ eta_bar - w_cap_prev @ y_bar_ap
        w_cap = w_cap_prev + e_bar_ap @ g_bar.T.conj()
        v_cap_hat = y_cap - g_bar @ (g_bar @ tau_bar).T.conj() @ y_cap
        return w_cap, z_cap, x_cap, v_cap_hat, e_bar_ap, g_bar


class Yast:
    """See Badeau et al., 2008"""

    @classmethod
    def track_spectral_weights(
        cls,
        x: npt.NDArray[float],
        w_cap_prev: npt.NDArray[complex],
        c_cap_xx_prev: npt.NDArray[complex],
        c_cap_yy_prev: npt.NDArray[complex],
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
            return w_cap_prev, c_cap_xx_prev, c_cap_yy_prev
        u = e / sigma
        x_ap = c_cap_xx_prev @ x
        y_ap = c_cap_yy_prev @ y
        y_apap = w_cap_prev.T.conj() @ x_ap
        # Added compared to Table 1
        c_cap_xx = beta * c_cap_xx_prev + x @ x.T.conj()
        #
        c_cap_yy_ap = beta * c_cap_yy_prev + y @ y.T.conj()
        z = beta * (y_apap - y_ap) / sigma + sigma * y
        gamma = sigma**2 + (beta / sigma**2) * (
            x.T.conj() @ x_ap - 2 * np.real(y.T.conj() @ y_apap) + y.T.conj() @ y_ap
        )
        # Filling $\bar{C}_{yy}(t)
        c_yy_bar = np.empty(c_cap_yy_ap.shape + tuple(np.ones(c_cap_yy_ap.ndim)))
        c_yy_bar[:-1, :-1] = c_cap_yy_ap
        c_yy_bar[:-1, -1] = z.T.conj()
        c_yy_bar[-1, :-1] = z
        c_yy_bar[-1, -1] = gamma
        #
        phi_var_bar, lambd = cls._track_conjugate_gradient(
            c_cap_yy_ap,
            c_yy_bar,
            z,
            track_principal=track_principal,
            nb_it=nb_it,
            thresh=thresh,
        )
        # Decomposition of vector phi_bar
        phi = np.abs(phi_var_bar[-1])
        assert phi <= 1, "Noooo"
        theta = phi_var_bar[-1] / phi
        epsilon = np.sqrt(1 - phi**2)
        phi_var = phi_var_bar[:-1] / (theta * epsilon)
        #
        e_1 = np.zeros_like(phi_var)
        e_1[0] = 1
        e_1 = -np.exp(1j * np.angle(phi_var[0])) * e_1
        #
        a = (phi_var - e_1) / alg.norm(phi_var - e_1, ord=None)
        b = w_cap_prev @ a
        q_cap = w_cap_prev - 2 * b @ a.T.conj() - epsilon * u @ e_1.T.conj()
        q_1 = q_cap[:, 0]
        d = np.ones_like(phi_var)
        d[0] = 1 / alg.norm(q_1, ord=None)
        d_cap = np.diag(d)
        w_cap = q_cap @ d_cap
        #
        c_cap_yy_ap_a = c_cap_yy_ap @ a
        a_ap = 4 * (c_cap_yy_ap_a - (a.T.conj() @ c_cap_yy_ap_a) @ a)
        z_ap = 2 * z - 4 * (a.T.conj() @ z) @ a - epsilon * gamma * e_1
        #
        c_cap_yy_apap = c_cap_yy_ap - a_ap @ a.T.conj() - epsilon * z_ap @ e_1.T.conj()
        c_cap_yy_apap = (c_cap_yy_apap + c_cap_yy_apap.T.conj()) / 2
        #
        c_cap_yy = d_cap @ c_cap_yy_apap @ d_cap
        return w_cap, c_cap_xx, c_cap_yy

    @staticmethod
    def _track_conjugate_gradient(
        c_cap_yy_ap: npt.NDArray[complex],
        c_cap_yy_bar: npt.NDArray[complex],
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
        p = c_cap_yy_ap @ g - (g.T.conj() @ c_cap_yy_ap @ g) @ g
        #
        s_cap = np.stack((np.zeros_like(g), p / alg.norm(p, ord=None), g), axis=1)
        s_cap = np.concatenate((s_cap, [1, 0, 0]), axis=0)
        #
        c_cap_ys_bar = c_cap_yy_bar @ s_cap
        c_cap_ss = s_cap.T.conj() @ c_cap_ys_bar
        assert c_cap_ss.shape == (3, 3)
        k = 0
        delta_j_cap = np.inf
        lambd = None
        theta = None
        while (k is None or k < nb_it) and (
            thresh is None or alg.norm(delta_j_cap) > thresh
        ):
            w, vr = alg.eig(c_cap_ss, left=False, right=True)
            w_norm = arg.norm(w)
            idx_extr = np.argmax(w_norm) if track_principal else np.argmin(w_norm)
            theta = w[idx_extr]
            lambd = vr[idx_extr]
            #
            theta_var = np.asarray(
                [
                    -alg.norm([theta[1], theta[2]], ord=2),
                    theta[0].conj()
                    * (theta[1] / np.abs(theta[1]))
                    / np.sqrt(1 + np.abs(theta[2] / theta[1]) ** 2),
                    theta[0].conj()
                    * (theta[2] / np.abs(theta[2]))
                    / np.sqrt(1 + np.abs(theta[1] / theta[2]) ** 2),
                ]
            )
            theta_cap = np.stack((theta, theta_var), axis=1)
            s_cap[:, :2] = s_cap @ theta_cap
            c_cap_ys_bar[:, :2] = c_cap_ys_bar @ theta_cap
            c_cap_ss[:2, :2] = theta_cap.T.conj() @ c_cap_ss @ theta_cap
            delta_j_cap = 2 * (c_cap_ys_bar[:, 0] - lambd * s_cap[:, 1])
            #
            g = delta_j_cap / alg.norm(delta_j_cap)
            g = g - s_cap[:, :2] @ (s_[:, :2].T.conj() @ g)
            g = g / alg.norm(g)
            #
            s_cap[:, 2] = g
            c_cap_ys_bar[:, 2] = c_cap_yy_bar @ s_cap[:, 2]
            c_cap_ss[:, 2] = s_cap.T.conj() @ c_ys_var[:, 2]
            c_cap_ss[2, :] = c_cap_ss[:, 3].T.conj()
        phi_bar = s_cap[:, 0]
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
        because then the best order is the argmax!

        Args:
            x (npt.NDArray[complex]): [description]
            n (int): [description]
            p_max (int): Maximum order to consider, in [|1, n-2|]

        Returns:
            List[npt.NDArray[complex]]: Estimation error of p for p in [1, p_max]
        """
        assert (
            1 <= p_max < n - 1
        ), f"Maximum order p_max={p_max} should be less than n-1={n-1}."

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
        e_cap = [np.empty((p, p), dtype=w_cap[0].dtype) for p in range(p_max + 1)]
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
            e_cap[p] = ksi_cap[p] - 1 / (1 - np.sum(np.abs(nu[p]) ** 2)) * np.outer(
                w_cap_down_p @ nu[p], phi[p].T.conj()
            )
        # discard p=0 case (meaningless)
        e_cap = e_cap[1:]
        return e_cap

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
        e_cap = cls.error(x, n, p_max)
        j_cap = np.asarray(
            [1 / alg.norm(e_cap[p], ord=None) ** 2 for p in range(len(e_cap))]
        )
        return j_cap

    @classmethod
    def estimate_esm_order(
        cls, x: npt.NDArray[complex], n: int, p_max: int, thresh_ratio: float = 0.1
    ) -> int:
        """Gets the estimated ESM model order r using the ESTER algorithm.
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
        j_cap = cls.inverse_error_func(x, n, p_max)
        j_max = np.amax(j_cap)
        # first select peaks in signal
        j_max_thres_ids, _ = sig.find_peaks(j_cap, height=j_max * thresh_ratio)
        # then filter peaks under threshold
        j_max_ids = sig.argrelextrema(j_cap, np.greater_equal, order=1, mode="clip")[0]
        j_max_thres_ids = j_max_ids[
            np.nonzero(j_cap[j_max_ids] >= j_max * thresh_ratio)[0]
        ]
        # first index corresponds to p=1, second to p=2 etc.
        r = np.amax(j_max_thres_ids) + 1
        return r


class AdaptiveEsprit:
    """Centralised methods used in the Adaptive ESPRIT framework."""

    @classmethod
    def estimate_esm(
        cls,
        x: npt.NDArray[float],
        n: int,
        r: int,
        l: int = None,
        log_progress: bool = False,
    ) -> Tuple[EsmModel, npt.NDArray[complex], npt.NDArray[complex]]:
        """Estimates the complete ESM model using the ESPRIT algorithm.

        Args:
            x (np.ndarray): input signal
            n (int): number of lines in the Hankel matrix S
            k (int): number of searched sinusoids

        Returns:
            Tuple[EsmModel, npt.NDArray[complex], npt.NDArray[complex]]: (EsmModel, Signal spectral matrix, Noise spectral matrix)
        """
        if l is None:
            # With respect to the Cramer-Rao bounds
            l = (3 * n) // 2
        n_cap = n + l - 1
        x_block = x[:n_cap]
        # FIRST RUN USING CLASSIC ESPRIT
        w_cap, _ = Esprit.subspace_weighting_mats(x_block, n, r)
        phi_cap = Esprit.spectral_matrix(w_cap)
        zs, g_cap = Esprit.estimate_poles(phi_cap)
        alphas = Esprit.estimate_esm_alphas(x_block, zs)
        esm = EsmModel.from_complex(zs, alphas)
        #
        psi_cap, _ = Esprit.partner_matrices(w_cap)
        #
        esm_list = [esm]
        #
        # See the various articles for initial values of the matrices
        # truncated window in the Fapi algorithm
        # p = 2
        # w_cap = np.concatenate((np.identity(r), np.zeros((n - r, r))), axis=0)
        z_cap = np.identity(r)
        x_cap = alg.hankel(x_block[:n], r=x_block[n - 1 :])
        v_cap_hat = np.zeros((r, l))
        # e = np.zeros((n))
        # g = np.zeros((r))
        # l samples overlap, step of 1
        # it = range(1, len(x) - n)
        nb_blocks = len(x) // n
        it = range(nb_blocks)
        if log_progress:
            it = tqdm.tqdm(it)
        for j in it:
            idx_start = j * n
            idx_stop = (j + 1) * n
            x_block = x[idx_start:idx_stop]
            w_cap_prev = w_cap
            z_cap_prev = z_cap
            w_cap, z_cap, e, g = Fapi.track_spectral_weights(
                x_block, w_cap_prev, z_cap_prev, beta=0.99
            )
            print(np.max(np.abs(z_cap)))
            """
            phi_cap, psi_cap, _, _ = MiscAdaptiveTracking.track_spectral_matrix_fae(
                e, g, w_cap, w_cap_prev, psi_cap, phi_cap
            )
            w_cap_id = w_cap.T.conj() @ w_cap
            w_cap_err = alg.norm(np.identity(r) - w_cap_id)
            # print(w_cap_err)
            zs, g_cap = Hrhatrac.track_poles(phi_cap, zs, g_cap)
            hey = np.max(np.abs(z_cap))
            # i_just_met_you = np.max(np.abs(zs))
            # _, and_this_is_crazy = EsmModel.poles_to_dampfreq(zs)
            # and_this_is_crazy = np.mean(np.abs(and_this_is_crazy))
            # print( "{:.2f} | {:.2f} | {:.2f}".format( hey, i_just_met_you, and_this_is_crazy))
            # The poles change at each step so the adaptive LS algorithm can't be used here.
            alphas = Esprit.estimate_esm_alphas(x_block, zs)
            #
            esm = EsmModel.from_complex(zs, alphas)
            # print(esm.nus * sr)
            #
            esm_list.append(esm)
            """
        esm_adapt = BlockEsmModel(esm_list)
        return esm_adapt


if __name__ == "__main__":
    sr = 4410
    n_s = 80000
    # number of sinusoids
    r = 8
    # adaptive stuff
    nb_blocks = 1
    l_block = n_s // nb_blocks
    # logging
    # log_progress = True
    log_progress = False
    # ESPRIT params
    n = 32
    l = (3 * n) // 2

    # Normalised damping ratios, multiply by sampling rate to get the 'deltas' in Amp.s-1
    gammas_list = np.repeat(rng.normal(0.002, 0.0001, (nb_blocks, r)), l_block, axis=0)
    # Normalised frequencies
    nus_list = np.repeat(rng.normal(0.1, 0.05, (nb_blocks, r)), l_block, axis=0)
    amps_list = np.repeat(rng.uniform(0.1, 1, (nb_blocks, r)), l_block, axis=0)
    phis_list = np.repeat(rng.uniform(0, 2 * np.pi, (nb_blocks, r)), l_block, axis=0)
    # Normalised damping ratios, multiply by sampling rate to get the 'deltas' in Amp.s-1
    gammas = rng.uniform(0.001, 0.01, r)
    # Normalised frequencies
    nus = rng.normal(0.1, 0.05, r)
    amps = rng.uniform(0.1, 1, r)
    phis = rng.uniform(0, 2 * np.pi, r)

    x_esm_adapt = BlockEsmModel.from_param_lists(
        gammas_list, nus_list, amps_list, phis_list
    )
    x_esm = EsmModel(gammas, nus, amps, phis)

    # one sample per value
    x_sine = x_esm.synth(n_s)
    x_sine = np.real(x_sine)
    wav.write("aaaa", data=np.real(x_sine) / np.max(np.real(x_sine)), rate=sr)

    x_esm_est = AdaptiveEsprit.estimate_esm(x_sine, n, r, l, log_progress=log_progress)

    # print(x_esm.nus[0] * sr)
    # print(x_esm_est.nus[0] * sr)
    # print(x_esm.gammas * sr)
    # print(x_esm_est.gammas * sr)

    # x_sine_est = x_esm_est.synth(n_s_block)
