import scipy
import numpy as np
from scipy import signal
from scipy.linalg import hankel
from scipy.linalg import toeplitz
from scipy.signal import lfilter

from numpy.linalg import pinv

import colorednoise as cn


def nextpow2(integer):
    power = 1
    while power < integer:
        power *= 2
    return power


def synthesize(
    length: int,
    delta: np.ndarray,
    freq: np.ndarray,
    amp: np.ndarray,
    phi: np.ndarray,
    f_m: float = 0,
    mod_index: float = 0,
    noise_index: int = 0,
    sin_to_noise: int = 0,
):
    """
    Synthesize a signal following the ESM, with a possible vibrato effect (FM) and/or noise.
    SNR can be controlled.

    Parameters
    ----------
    `length`: length of the signal (discrete)
    `delta`: array of damping factors
    `freq`: array of normalized frequencies
    `amp`: array of real amplitudes
    `phi`: array of initial phases
    `f_m`: modulating frequency (FM)
    `mod_index`: modulation index (0 if vibrato is not desired)
    `noise_index`:  0 = noiseless; 1 = white noise; 2 = pink noise; 3 = red noise
    `sin_to_noise`: sinusoids to noise ratio to estimate a desired SNR

    Return
    ------
    `sig`: synthesized signal
    `sinusoids`: pure sinusoidal part (complexe)
    `noise`: noise part
    `snr`: signal-to-noise ratio
    """
    t_array = np.arange(length)  # time array
    logz = delta + 1j * 2 * np.pi * freq  # log of poles
    alpha = amp * np.exp(1j * phi)  # complex amplitudes

    mod = np.exp(
        1j * mod_index * np.sin(2 * np.pi * f_m * t_array)
    )  # modulation component

    sinusoids = np.sum(
        np.outer(alpha, np.ones(length)) * np.exp(np.outer(logz, t_array)), axis=0
    )  # noisless signal (ESM)
    sinusoids = sinusoids * mod  # modulated signal

    if noise_index == 0:
        noise = np.zeros(length)
        snr = np.inf
        sig = sinusoids

    else:
        noise = cn.powerlaw_psd_gaussian(noise_index - 1, length)  # synthesizing noise

        # normalaizing the noise according to a sinusoids-to-noise ratio (in dB)
        norm = np.exp(
            -(1 / 20)
            * (
                sin_to_noise
                + 10 * np.log10(np.sum(noise**2))
                - 10 * np.log10(np.sum(np.real(sinusoids) ** 2))
            )
        )
        noise = norm * noise
        sig = sinusoids + noise
        snr = 10 * np.log10(np.sum(np.real(sig) ** 2)) - 10 * np.log10(
            np.sum(noise**2)
        )

    return sig, sinusoids, noise, snr


def preemphasize(sig: np.ndarray, b=np.array([1, -0.95])):
    """
    Pre-emphasize the signal with an FIR of order 2.

    Parameters
    ----------
    `sig`: the input signal
    `b`: coefficients of the filter's transfer function

    Return
    ------
    `preemph_sig`: pre-emphasized signal
    `freq`: normalized frequencies
    `psd_sig`: PSD of the input signal
    `psd_preemph_sig`: PSD of the pre-emphasized signal
    """
    N = len(sig)
    Nfft = nextpow2(N)

    a = np.ones(2)
    preemph_sig = signal.lfilter(b, a, sig)

    freq, psd_sig = signal.welch(sig, nfft=Nfft)
    _, psd_preemph_sig = signal.welch(preemph_sig, nfft=Nfft)

    return preemph_sig, freq, psd_sig, psd_preemph_sig


def filter_bank(
    num_bands: int, fs: int, filter_length: int = 425, trans_width: int = 425
):
    """Create a filter bank with Remez' algotithm.  Parameters ---------- `sig`: input signal `num_bands`: number of sub-bands `fs`: sampling rate `filter_length`: number of taps in each filter = order of filter + 1 (choose it odd for the hp filter to function well) `trans_width`: transition width (in Hz)

    Return
    ------
    `coeffs`: an array containing the numpy arrays of the channel's coefficients
    """
    band_length = int(0.5 * fs / num_bands)

    # calculating band edges
    band_lp = [
        0,
        band_length,
        band_length + trans_width,
        0.5 * fs,
    ]  # first band for a lowpass filter
    amp_lp = [num_bands, 0]  # corresponding desired amplitudes

    coeffs_lp = signal.remez(filter_length, band_lp, amp_lp, Hz=fs)

    coeffs = [coeffs_lp]

    for i in range(1, num_bands - 1):
        band_bp = [
            0,
            i * band_length - trans_width,
            i * band_length,
            (i + 1) * band_length,
            (i + 1) * band_length + trans_width,
            0.5 * fs,
        ]  # ith band for a bandpass filter
        amp_bp = [0, num_bands, 0]  # corresponding desired amplitudes

        coeffs_bp = signal.remez(filter_length, band_bp, amp_bp, Hz=fs)

        coeffs.append(coeffs_bp)

    band_hp = [
        0,
        (num_bands - 1) * band_length - trans_width,
        (num_bands - 1) * band_length,
        0.5 * fs,
    ]  # last band for a highpass filter
    amp_hp = [0, num_bands]  # corresponding desired amplitudes

    coeffs_hp = signal.remez(filter_length, band_hp, amp_hp, Hz=fs)

    coeffs.append(coeffs_hp)

    return coeffs


def decimate(sig: np.ndarray, factor: int):
    """
    Decimate a signal.

    Parameters
    ----------
    `sig`: input signal (must be filtered first according to the decimation factor)
    `factor`: decimation factor

    Return
    ------
    `decimated`: decimated signal
    """
    N = len(sig)
    decimated = np.zeros(N // factor)

    for i in range(len(decimated)):
        decimated[i] = sig[i * factor]

    return decimated


def whiten(sig: np.ndarray, smoothing_order: int, rank: int, ar_order: int):
    """
    Whiten a signal with colored noise.

    Parameters
    ----------
    `sig`: the input signal
    `smoothing_order`: (discrete) at least two times the length of the PSD's principal lobe
    `rank` : rank factor; for example 2 for median, 3 for (1/3)-(2/3), etc...
    `ar_order`: autoregressive model order (10 to 20)

    Return
    ------
    `sig_white`: the filtered signal
    `freq`: frequencies for the PSD's
    `psd_sig`: PSD of the input signal
    `psd_noise`: Estimation of the colored noise PSD
    `psd_sig_white`: PSD of the filtered signal
    `psd_noise_white`: Estimatiobn of the white noise PSD
    """
    N = len(sig)
    Nfft = nextpow2(N)

    # Step 1: signal's power spectral density
    freq, psd_sig = signal.welch(sig, nfft=Nfft)  # return_onesided = False
    # PSD_x, freq = plt.psd(x, Nfft)

    # Step 2: estimating the noise's PSD with a median filter (smoothing the signal's PSD)
    psd_noise = scipy.ndimage.rank_filter(
        psd_sig, smoothing_order // rank, smoothing_order
    )  # smoothed PSD = estimation of the noise PSD

    # Step 3: calculating the autocovariance of the noise
    AC = np.real(np.fft.ifft(psd_noise, Nfft))  # autocovariance (vector) of the noise
    print(AC)
    R = toeplitz(
        AC[: ar_order - 1], AC[: ar_order - 1]
    )  # coefficients matrix of the Yule-Walker system
    # = autocovariance matrix with the last row and last column removed
    r = AC[1:ar_order].T  # the constant column of the Yule-Walker system
    B = -pinv(R) @ r  # the AR coefficients (indices 1, ..., N-1)
    B = np.insert(B, 0, 1)  # the AR coefficients (indices 0, ..., N-1)

    # Step 4: applying the corresponding FIR to the signal's PSD to obtain the whitened signal
    # The FIR is the inverse of the AR filter so the coefficients of the FIR's numerator
    # are the coefficients of the AR's denominator, i.e. the array B
    # denominator co-eff of the FIR's transfer function is 1
    sig_white = signal.lfilter(B, [1], sig)

    # Step 5: re-estimating the noise, now for the white signal
    _, psd_sig_white = signal.welch(sig_white, nfft=Nfft)
    # PSD_x_white, freq = plt.psd(x_white, Nfft)
    psd_noise_white = scipy.ndimage.rank_filter(
        psd_sig_white, smoothing_order // rank, smoothing_order
    )

    return sig_white, freq, psd_sig, psd_noise, psd_sig_white, psd_noise_white


def esprit(sig: np.ndarray, n: int, K: int):
    """
    Calculate the damping factors and frequencies by ESPRIT method.

    Parameters
    ----------
    `sig`: input signal
    `n`: number of lines in the Hankel matrix S
    `K`: signal space dimension = number of sinusoids (n-K : noise space dimension)

    Return
    ------
    `delta`: array of damping factors
    `freq`: array of normalized frequencies
    """
    N = len(sig)  # signal's length
    l = N - n + 1  # number of columns of the Hankel matrix
    # Not needed for the 'hankel' function but used in the formula of R_XX

    X = hankel(sig[:n], sig[n - 1 :])  # Hankel matrix

    R_XX = 1 / l * X @ X.conj().T  # correlation matrix
    U1, Lambda, U2 = np.linalg.svd(R_XX)
    W = U1[:n, :K]  # signal space matrix

    W_down = W[:-1]
    W_up = W[1:]
    phi = (np.linalg.pinv(W_down)) @ W_up
    eigenvalues, eigenvectors = np.linalg.eig(phi)

    delta = np.log(np.abs(eigenvalues))  # damping factors
    freq = (1 / (2 * np.pi)) * np.angle(eigenvalues)  # frequencies

    return delta, freq


def least_squares(sig: np.ndarray, delta: np.ndarray, freq: np.ndarray):
    """
    Calculate the amplitudes and phases with Least Squares method.

    Parameters
    ----------
    `sig`: input signal
    `delta`: array of damping factors
    `freq`: array of frequencies

    Return
    ------
    `alpha`: array of complex amplitudes
    `amp`: array of real amplitudes
    `phi`: array of initial phases

    """
    N = len(sig)  # signal's length
    t = np.arange(N)  # array of discrete times
    s = delta + 2j * np.pi * freq  # log of the pole
    VN = np.exp(np.outer(t, s))  # Vandermonde matrix of dimension N

    alpha = np.linalg.pinv(VN) @ sig
    amp = abs(alpha)
    phi = np.angle(alpha)

    return alpha, amp, phi


def energy(sig: np.ndarray, delta: np.ndarray, amp: np.ndarray):
    """
    Calculate the energies of sinusoids.

    Parameters
    ----------
    `sig`: input signal
    `delta`: array of damping factors
    `amp`: array of real amplitudes

    Return
    ------
    `energy_db`: energy array in dB
    """
    N = len(sig)  # signal's length
    times = np.arange(N)  # array of discrete times

    K = len(delta)  # number of sinusoids
    E = np.zeros(K)

    for k in range(K):  # calculating the energy of each sinusoid
        e_k = 0
        for t in times:
            e_k += np.exp(2 * delta[k] * t)  # the contribution of delta
        E[k] = amp[k] ** 2 * e_k  # the energy of the kth sinusoid

    Emax = max(E)
    energy_db = 10 * np.log(E / Emax)  # Energy in dB

    return energy_db


def esprit_blocks(
    sig: np.ndarray,
    window_length: float,
    hop_size: float,
    sampling_rate: int,
    n: int,
    K: int,
):
    """
    Compute `esprit` and `least_square` by blocks.

    Parameters
    ----------
    `sig`: the full-length input signal
    `window_length`: the window size (in seconds)
    `hop_size`: the hop size (in seconds)
    `n`: number of lines in the Hankel matrix S
    `K`: signal space dimension = number of sinusoids (n-K: noise space dimension)

    Return
    ------
    (for blocks by rows)
    `delta`: matrix of damping factors
    `freq`: matrix of normalized frequencies
    `amp`: matrix of real amplitudes
    `phi`: matrix of initial phases
    """
    N_sig = len(sig)  # the length of the signal (in samples)
    N_block = int(window_length * sampling_rate)  # window size (in samples)
    h = int(hop_size * sampling_rate)  # hop size (in samples)
    I = int((N_sig - N_block) / h)

    delta = np.zeros((I, K))
    freq = np.zeros((I, K))
    alpha = np.zeros((I, K), dtype=complex)
    amp = np.zeros((I, K))
    phi = np.zeros((I, K))
    # energy_db = np.zeros((I,K))

    for i in range(I):

        sig_i = sig[h * i : h * i + N_block]  # ith truncated signal

        delta[i], freq[i] = esprit(sig_i, n, K)
        alpha[i], amp[i], phi[i] = least_squares(sig_i, delta[i], freq[i])
        # energy_db[i] = Energy(sig[i], delta[i], amp[i])

    return delta, freq, alpha, amp, phi  # , energy_db
