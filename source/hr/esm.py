import numpy as np
import numpy.typing as npt


class EsmModel:
    """
    Exponential sinusoidal model
    Sinusoids are sorted by increasing frequencies
    """

    def __init__(
        self,
        gammas: npt.NDArray[float],
        nus: npt.NDArray[float],
        amps: npt.NDArray[float],
        phis: npt.NDArray[float],
    ) -> None:
        """[summary]

        Args:
            gammas (npt.NDArray[float]): Normalised damping (multiply by sampling rate to get 'delta' in Amp.s-1)
            nus (npt.NDArray[float]): Normalised frequencies, in [-0.5, 0.5]
            amps (npt.NDArray[float]): [description]
            phis (npt.NDArray[float]): [description]
        """
        gammas = np.array(gammas)
        nus = np.array(nus)
        amps = np.array(amps)
        phis = np.array(phis)
        r = gammas.shape[0]
        # Some safety checks
        s = gammas.shape
        assert (
            gammas.ndim == 1 and nus.shape == s and amps.shape == s and phis.shape == s
        ), "Inconsistent argument shape."
        assert np.all(gammas >= 0), "Dampings should be positive or null."
        # assert np.all(-0.5 <= nus) and np.all(nus <= 0.5), "Frequencies are normalised (reduced)."
        # sort by increasing frequencies
        incr_ids = np.argsort(nus)
        gammas = np.take_along_axis(gammas, incr_ids, axis=-1)
        nus = np.take_along_axis(nus, incr_ids, axis=-1)
        amps = np.take_along_axis(amps, incr_ids, axis=-1)
        phis = np.take_along_axis(phis, incr_ids, axis=-1)
        self.gammas = gammas
        self.nus = nus
        self.amps = amps
        self.phis = phis
        self.r = r

    def synth(
        self,
        n_s: int,
    ):
        """Synthesizes a sinusoidal signal, with set frequencies,
        amplitudes, initial phases,
        and damping factors.

        The frequencies $\nu_k$ and damping factors $\gamma_k$ should
        be those of a signal sampled at rate 1. We will call them 'normalised'.

        Args:
            ts (npt.NDArray[float]): Times array (seconds)
            gammas (npt.NDArray[float]): Normalised damping factors array
            nus (npt.NDArray[float]): Normalised frequencies array (in [0, 0.5])
            amps (npt.NDArray[float]): Real amplitudes
            phi (npt.NDArray[float]): Initial phases

        Returns:
            [type]: [description]
        """
        ts = np.arange(n_s)  # discrete times
        log_zs = -self.gammas + 1j * 2 * np.pi * self.nus  # log of poles
        alphas = self.amps * np.exp(1j * self.phis)  # complex amplitudes

        x = np.sum(
            np.outer(alphas, np.ones_like(ts)) * np.exp(np.outer(log_zs, ts)), axis=0
        )  # noisless signal (ESM)

        return x

    @staticmethod
    def make_fm_mod(
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

    @staticmethod
    def make_noisy(
        x_sine: npt.NDArray[complex],
        mod: npt.NDArray[float],
        noise: npt.NDArray[float],
        sine_to_noise: float,
    ) -> npt.NDArray[complex]:
        """Combines everything

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
