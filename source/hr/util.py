import scipy.linalg
import scipy.signal as sig
import numpy as np
import numpy.typing as npt

from typing import Tuple, List

# by Robert Bristow-Johnson
# taken from: https://www.firstpr.com.au/dsp/pink-noise/
_PINK_FILTRE_COEFFS = {
    "b": [0.98443604, 0.83392334, 0.07568359],
    "a": [0.99572754, 0.94790649, 0.53567505],
}


def esprit_correlation_mat(x: npt.NDArray[complex], n: int) -> npt.NDArray[complex]:
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


def esprit_ws(
    x: npt.NDArray[complex], n: int, k: int
) -> Tuple[npt.NDArray[complex], npt.NDArray[float]]:
    """Gets W and W_per used in ESPRIT algorithm

    Args:
        x (npt.NDArray[complex]): [description]
        n (int): [description]
        k (int): [description]

    Returns:
        Tuple[npt.NDArray[complex], npt.NDArray[float]]: (n,k) and (n,n-k) W and W_per matrices
    """
    r_xx = esprit_correlation_mat(x, n)
    u_1, _, _ = scipy.linalg.svd(r_xx)
    # sigma = scipy.linalg.diagsvd(s, r_xx.shape[0], r_xx.shape[1])
    w = u_1[:, :k]
    w_per = u_1[:, k:]
    return w, w_per


def esprit(
    x: npt.NDArray[float], n: int, k: int
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
    w, _ = esprit_ws(x, n, k)

    w_down = w[:-1]
    w_up = w[1:]
    phi = scipy.linalg.pinv(w_down) @ w_up
    zs = scipy.linalg.eig(phi, left=False, right=False)
    deltas = -np.log(np.abs(zs))  # damping factors
    nus = np.angle(zs) / (2 * np.pi)  # frequencies
    return deltas, nus


def least_squares(
    x: npt.NDArray[float], deltas: npt.NDArray[float], nus: npt.NDArray[float]
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


def synth_sine(
    n_s: int,
    deltas: npt.NDArray[float],
    nus: npt.NDArray[float],
    amps: npt.NDArray[float],
    phis: npt.NDArray[float],
):
    """Synthesizes a sinusoidal signal, with set frequencies,
    amplitudes, initial phases,
    and damping factors.

    The frequencies $\nu_k$ and damping factors $\delta_k$ should
    be those of a signal sampled at rate 1. We will call them 'normalised'.

    Args:
        ts (npt.NDArray[float]): Times array (seconds)
        deltas (npt.NDArray[float]): Normalised damping factors array
        nus (npt.NDArray[float]): Normalised frequencies array (in [0, 0.5])
        amps (npt.NDArray[float]): Real amplitudes
        phi (npt.NDArray[float]): Initial phases

    Returns:
        [type]: [description]
    """
    ts = np.arange(n_s)  # discrete times
    log_zs = -deltas + 1j * 2 * np.pi * nus  # log of poles
    alphas = amps * np.exp(1j * phis)  # complex amplitudes

    x = np.sum(
        np.outer(alphas, np.ones_like(ts)) * np.exp(np.outer(log_zs, ts)), axis=0
    )  # noisless signal (ESM)

    return x


def mod_fm(
    n_s: int, nus: npt.NDArray[float], amps: npt.NDArray[float]
) -> npt.NDArray[complex]:
    """Make modulation component in FM synthesis

    Args:
        n_s (int): Number of temporal samples
        nus (npt.NDArray[float]): Normalised frequencies array (in [0, 0.5])
        amps (npt.NDArray[float]): Real amplitudes array

    Returns:
        npt.NDArray[complex]: Modulation component to be used in FM synthesis
    """
    ts = np.arange(n_s)
    return np.exp(1j * amps * np.sin(2 * np.pi * nus * ts))


def make_synth(
    x_sine: npt.NDArray[complex],
    mod: npt.NDArray[float],
    noise: npt.NDArray[float],
    sine_to_noise: float,
) -> npt.NDArray[complex]:
    """Combine

    Args:
        x_synth (npt.NDArray[complex]): [description]
        mod (npt.NDArray[float]): [description]
        noise (npt.NDArray[float]): [description]
        sine_to_noise (float): Blending ratio of harmonic and noise parts.

    Returns:
        npt.NDArray[complex]: Synthesied modulated then noised harmonic signal
    """
    x_mod = x_sine * mod

    # normalising the noise according to a sinusoids-to-noise ratio (in dB)
    norm_noise = np.exp(
        -(1 / 20)
        * (
            sine_to_noise
            + 10 * np.log10(np.sum(noise**2))
            - 10 * np.log10(np.sum(np.real(x_mod) ** 2))
        )
    )
    noise_normed = norm_noise * noise
    x_synth = x_mod + noise
    snr = 10 * np.log10(np.sum(np.real(x_synth) ** 2)) - 10 * np.log10(
        np.sum(noise_normed**2)
    )

    return x_synth, snr


def psd_noise(x_psd: npt.NDArray[complex], nu_width: float) -> npt.NDArray[complex]:
    """Estimates the noise's PSD with a median filter (smoothing the signal's PSD)
    Args:
        x_psd (npt.NDArray[complex]): [description]
        nu_width (float): Width of the median filter in normalised frequency (in [0, 0.5])

    Returns:
        npt.NDArray[complex]: Estimated PSD of the noise.
    """
    n_fft = x_psd.shape[-1]
    size = int(np.round(nu_width * n_fft))

    noise_psd = scipy.ndimage.median_filter(x_psd, size=size)
    return noise_psd


def noise_filtre_coeffs(
    x: npt.NDArray[complex], n_fft: int, nu_width: float, ar_ordre: int
) -> npt.NDArray[complex]:
    """Estimate the coefficients of the filtre generating the coloured noise from a white one.

    Args:
        x (npt.NDArray[complex]): Input temporal signal
        n_fft (int): Number of frequential bins
        nu_width (float): Width of the median filtre used in noise PSD estimation
        ar_ordre (int): Ordre of the estimated AR filtre

    Returns:
        npt.NDArray[complex]: Coefficients of the AR filtre.
    """
    # x: the input signal FOR EACH FREQUENCY BAND
    # smoothing_ordre: at least two times the length of the PSD's principal lobe (can be done visually)
    # AR_ordre: ~ 10
    # note: no need to give it the correct sample frequency,
    # since it is only used to return the fftfreqs.
    _, x_psd = sig.welch(x, fs=1, nfft=n_fft)

    # Step 2: estimating the noise's PSD with a median filtre (smoothing the signal's PSD)
    noise_psd = psd_noise(x_psd, nu_width)

    # Step 3: calculating the autocovariance of the noise
    # autocovariance (vector) of the noise
    ac_coeffs = np.real(np.fft.ifft(noise_psd, n=n_fft))
    # coefficients matrix of the Yule-Walker system
    # = autocovariance matrix with the last row and last column removed
    r_mat = scipy.linalg.toeplitz(ac_coeffs[: ar_ordre - 1], ac_coeffs[: ar_ordre - 1])
    # the constant column of the Yule-Walker system
    r = ac_coeffs[1:ar_ordre].T
    # the AR coefficients (indices 1, ..., N-1)
    b = -scipy.linalg.pinv(r_mat) @ r
    # the AR coefficients (indices 0, ..., N-1)
    b = np.insert(b, 0, 1)

    return b


def whiten(
    x: npt.NDArray[complex], n_fft: int, nu_width: float, ar_ordre: int
) -> npt.NDArray[complex]:
    """Whiten the noise in input temporal signal

    Args:
        x (npt.NDArray[complex]): Input temporal signal
        n_fft (int): Number of frequential bins
        nu_width (float): Width of the median filtre used in noise PSD estimation
        ar_ordre (int): Ordre of the estimated AR filtre

    Returns:
        npt.NDArray[complex]: Temporal signal with noise whitened.
    """
    b = noise_filtre_coeffs(x, n_fft=n_fft, nu_width=nu_width, ar_ordre=ar_ordre)
    # Step 4: applying the corresponding FIR to the signal's PSD to obtain the whitened signal
    # The FIR is the inverse of the AR filter so the coefficients of the FIR's numerator
    # are the coefficients of the AR's denominator, i.e. the array b
    # denominator coeff of the FIR's transfer function is 1
    x_whitened = sig.lfilter(b, [1], x)
    return x_whitened


def ester_error(
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

    w, _ = esprit_ws(x, n, p_max)
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
    e = [np.zeros((p, p), dtype=w_cap[0].dtype) for p in range(p_max + 1)]
    for p in range(1, p_max + 1):
        # Recursive computation of e[p]
        # see Badeau et al. (2006) Table 1 for details on the algorithm
        w_cap_down_pm = w_cap[p - 1][:-1]
        w_cap_up_pm = w_cap[p - 1][1:]
        w_down_p = w_cap[p][:-1, p - 1]
        w_up_p = w_cap[p][1:, p - 1]
        # print(w_down_p.shape, w_up_p.shape)
        # 1. Update of the auxiliary matrix psi_mat[p]
        psi_r[p] = w_cap_down_pm.T.conj() @ w_up_p
        psi_l[p] = w_cap_up_pm.T.conj() @ w_down_p
        psi_lr[p] = w_down_p.T.conj() @ w_up_p
        # print(w_cap_up_pm.shape, w_down_p.shape)
        # 2. Update of the auxiliary matrix ksi_mat[p]
        # print(w_up_p.shape)
        # print(w_cap_down_pm.shape, psi_r[p].shape)
        # print(w_down_p.shape, psi_lr[p].shape)
        ksi_p = w_up_p - w_cap_down_pm @ psi_r[p] - w_down_p * psi_lr[p]
        # print(ksi_cap[p - 1].shape)
        # print(ksi_cap[p][:, :-1].shape)
        psi_l_p = psi_l[p]
        ksi_cap[p][:, :-1] = ksi_cap[p - 1] - np.outer(w_down_p, psi_l_p.conj())
        ksi_cap[p][:, -1] = ksi_p
        # 3. Computation of e[p] from ksi_cap[p]
        mu_p = nu[p][-1]
        phi[p][:-1] = phi[p - 1] + mu_p * psi_l[p]
        phi[p][-1] = psi_r[p].T.conj() @ nu[p - 1] + mu_p * psi_lr[p].conj()
        w_cap_down_p = w_cap[p][:-1]
        a = w_cap_down_p @ nu[p]
        e[p] = ksi_cap[p] - 1 / (1 - np.linalg.norm(nu[p], ord=2) ** 2) * np.outer(
            (w_cap_down_p @ nu[p]), phi[p].T.conj()
        )
    # discard p=0 bogus case.
    e = e[1:]
    return e


def ester_inverse_error_func(
    x: npt.NDArray[complex], n: int, p_max: int
) -> npt.NDArray[float]:
    """Gets the inverse error function J in the ESTER algorithm

    Args:
        x (npt.NDArray[complex]): [description]
        n (int): [description]
        p_max (int): [description]

    Returns:
        npt.NDArray[float]: [description]
    """
    e = ester_error(x, n, p_max)
    j = np.array([1 / np.linalg.norm(e[p], ord=2) ** 2 for p in range(len(e))])
    return j


def ester(x: npt.NDArray[complex], n: int, p_max: int) -> npt.NDArray[float]:
    """Gets the estimated ESM model ordre r using the ESTER algorithm.
    see: http://ieeexplore.ieee.org/document/1576975/

    Args:
        x (npt.NDArray[complex]): [description]
        n (int): [description]
        p_max (int): [description]

    Returns:
        npt.NDArray[float]: [description]
    """
    j = ester_inverse_error_func(x, n, p_max)
    # first index actually corresponds to p=1
    r = np.argmax(j) + 1
    return r
