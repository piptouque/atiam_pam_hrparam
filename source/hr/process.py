from typing import Tuple, List, Union

import scipy.linalg as alg
import scipy.signal as sig
import scipy.ndimage as img
import numpy as np

rng = np.random.default_rng(123)
import numpy.typing as npt


from source.hr.esm import EsmModel, AdaptiveEsmModel
from source.hr.preprocess import NoiseWhitening, FiltreBank


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
        r_xx = cls._correlation_mat(x, n)
        u_1, _, _ = alg.svd(r_xx)
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
    def estimate_poles(
        cls,
        phi_cap: npt.NDArray[complex],
    ) -> npt.NDArray[complex]:
        """Estimates the poles of the spectral matrix

        Args:
            phi_cap (npt.NDArray[complex]): Spectral matrix

        Returns:
            npt.NDArray[float], npt.NDArray[float]]: (estimated normalised frequencies, estimated normalised dampings)
        """
        zs = alg.eig(phi_cap, left=False, right=False)
        return zs

    @classmethod
    def estimate_esm_alphas(
        cls,
        x: npt.NDArray[float],
        zs: npt.NDArray[complex],
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
        # signal's length
        n_s = len(x)
        ts = np.arange(n_s)  # array of discrete times
        # Vandermonde matrix of dimension N
        v_mat = np.exp(np.outer(ts, log_zs))
        alphas = alg.pinv(v_mat) @ x
        return alphas

    @classmethod
    def estimate_esm(
        cls, x: npt.NDArray[float], n: int, r: int
    ) -> Tuple[EsmModel, npt.NDArray[complex], npt.NDArray[complex]]:
        """Estimates the complete ESM model using the ESPRIT algorithm.

        Args:
            x (np.ndarray): input signal
            n (int): number of lines in the Hankel matrix S
            r (int): number of searched sinusoids

        Returns:
            Tuple[EsmModel, npt.NDArray[complex], npt.NDArray[complex]]: (EsmModel, Signal spectral matrix, Noise spectral matrix)
        """
        w_cap, w_cap_per = cls.subspace_weighting_mats(x, n, r)
        phi_cap = cls.spectral_matrix(w_cap)
        zs = cls.estimate_poles(phi_cap)
        alphas = cls.estimate_esm_alphas(x, zs)
        #
        esm = EsmModel.from_complex(zs, alphas)

        return (esm, w_cap, w_cap_per)


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
        # Init values of interest from parameters
        nu = w_cap[-1].T.conj()
        nu_norm_sq = np.sum(np.abs(nu) ** 2)
        w_cap_down_prev, w_up_prev = w_cap_prev[:-1], w_cap_prev[1:]
        e_down, e_up = e[:-1], e[1:]
        # Algorithm given in Table 1
        e_minus = w_cap_down_prev.T.conj() @ e_up
        e_plus = w_up_prev.T.conj() @ e_down
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
        # Additional stuff
        nu_prev = w_cap_prev[-1].T.conj()
        nu_prev_norm_sq = np.sum(np.abs(nu_prev) ** 2)
        e_n = e[-1]
        #
        phi_prev = psi_cap_prev.conj().H @ nu_prev
        e_plus_apap = e_plus_ap + (e_n / (1 - nu_norm_sq)) * phi
        delta_phi = phi / (1 - nu_norm_sq) - phi_prev / (1 - nu_prev_norm_sq)
        #
        a_bar = np.stack((g, e_minus, nu_prev), axis=0)
        b_bar = np.stack((e_plus_apap, g, delta_phi), axis=0)
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
        raise NotImplementedError()
        #
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

    @staticmethod
    def track_poles_hrhatrac(
        phi_cap: npt.NDArray[complex],
        d_prev: npt.NDArray[complex],
        g_cap_prev: npt.NDArray[complex],
        mu_d: float = 0.99,
        mu_g: float = 0.99,
    ) -> Tuple[npt.NDArray[complex], npt.NDArray[complex]]:
        """
        and HrHatrac David et al., 2006
        For reference. Same notations as the article.

        Args:
            g_cap_prev (npt.NDArray[complex]): [description]
            g_cap_ap_prev (npt.NDArray[complex]): [description]
            d_prev (npt.NDArray[complex]): Previous eigenvalues as a vector
            a_bar (npt.NDArray[complex]): [description]
            b_bar (npt.NDArray[complex]): [description]

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
        # TODO: normalise the columns of $V(t)$ for numerical stability
        g_cap = g_cap / alg.norm(g_cap, axis=1, keepdims=True)
        return d, g_cap


class Fls:
    """Fast sequential LS estimation ...
    see David et Badeau, 2007
    """

    @staticmethod
    def track_esm_alphas(
        x: npt.NDArray[float],
        d: npt.NDArray[complex],
    ) -> npt.NDArray[complex]:
        raise NotImplementedError()


class Fapi:
    """See Badeau et al., 2005"""

    @staticmethod
    def track_spectral_weights_fapi(
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
    def track_spectral_weights_swfapi(
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
            w_cap_prev (npt.NDArray[complex]): [description]
            z_cap_prev (npt.NDArray[float]): [description]
            x_cap_prev (npt.NDArray[float]): [description]
            v_cap_hat_prev (npt.NDArray[float]): [description]
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
        # r = w_cap_prev.shape[-1]
        # Rank of the update involved in equation (4)
        # p = 2 in the truncated window case.
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
            tau_bar @ np.inv(eta_bar)
        ).T.conj()
        z_cap = (1 / beta) * (
            z_cap_prev - g_bar @ h_bar_ap.T.conj() + epsilon_bar @ g_bar.T.conj()
        )
        # an n x p matrix
        e_bar_ap = x_bar @ eta_bar - w_cap_prev @ y_bar_ap
        w_cap = w_cap_prev + e_bar_ap @ g_bar.T.conj()
        v_cap_hat = y_cap - g_bar(g_bar @ tau_bar).T.conj() @ y_cap
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
        print(c_cap_ss.shape)
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
        cls, x: npt.NDArray[float], n: int, r: int, l_block: int, l_win: int = None
    ) -> Tuple[EsmModel, npt.NDArray[complex], npt.NDArray[complex]]:
        """Estimates the complete ESM model using the ESPRIT algorithm.

        Args:
            x (np.ndarray): input signal
            n (int): number of lines in the Hankel matrix S
            k (int): number of searched sinusoids

        Returns:
            Tuple[EsmModel, npt.NDArray[complex], npt.NDArray[complex]]: (EsmModel, Signal spectral matrix, Noise spectral matrix)
        """
        if l_win is None:
            l_win = 120
        nb_blocks = x.shape[0] // l_block + 1
        #
        x_block = x[:l_block]
        # FIRST RUN USING CLASSIC ESPRIT
        w_cap, w_cap_per = Esprit.subspace_weighting_mats(x_block, n, r)
        phi_cap = Esprit.spectral_matrix(w_cap)
        zs = Esprit.estimate_poles(phi_cap)
        alphas = Esprit.estimate_esm_alphas(x, zs)
        esm = EsmModel.from_complex(zs, alphas)
        #
        esm_list = [esm]
        w_cap_list = [w_cap]
        #
        # See the various articles for initial values of the matrices
        p = 2  # truncated window in the Fapi algorithm
        z_cap = np.identity(r)
        x_cap = np.zeros((l_block, l_win))
        v_cap_hat = np.zeros((r, l_win))
        e = np.zeros((l_block, p))
        g = np.zeros((r, p))
        #
        for j in range(1, nb_blocks):
            x_block = x[j * l_block : (j + 1) * l_block]
            w_cap, z_cap, x_cap, v_cap_hat, e, g = Fapi.track_spectral_weights_swfapi(
                x_block, w_cap, z_cap, x_cap, v_cap_hat
            )
            phi_cap, psi_cap = MiscAdaptiveTracking.track_spectral_matrix(
                e, g, w_cap, w_cap, psi_cap
            )
            zs, g_cap = MiscAdaptiveTracking.track_eigen_hrhatrac(phi_cap, zs, g_cap)
            # The poles change at each step so the adaptive LS algorithm can't be used here.
            alphas = Esprit.estimate_esm_alphas(x_block, zs)
            #
            esm = EsmModel.from_complex(zs, alphas)
            #
            esm_list.append(esm)
            w_cap_list.append(w_cap)
        return esm, w_cap, None


if __name__ == "__main__":
    sr = 44100
    n_s_block = 512
    n_fft = 1024

    nb_blocks = 1200
    # number of sinusoids
    r = 8
    # Normalised damping ratios, multiply by sampling rate to get the 'deltas' in Amp.s-1
    gammas_list = rng.normal(0.002, 0.0001, (nb_blocks, r))
    # Normalised frequencies
    nus_list = rng.normal(0.1, 0.05, (nb_blocks, r))
    amps_list = rng.uniform(0.1, 1, (nb_blocks, r))
    phis_list = rng.uniform(0, 2 * np.pi, (nb_blocks, r))

    x_esm = AdaptiveEsmModel.from_param_lists(
        gammas_list, nus_list, amps_list, phis_list
    )

    x_sine = x_esm.synth(n_s_block)

    n_est = 20
    x_esm_est, _, _ = AdaptiveEsprit.estimate_esm(x_sine, n_est, r, n_s_block)

    # print(x_esm.nus * sr)
    # print(x_esm_est.nus * sr)
    print(x_esm.gammas * sr)
    # print(x_esm_est.gammas * sr)

    x_sine_est = x_esm_est.synth(n_s)
