import numpy as np
import numpy.typing as npt


# by Robert Bristow-Johnson
# taken from: https://www.firstpr.com.au/dsp/pink-noise/
_PINK_FILTRE_COEFFS = {
    "b": [0.98443604, 0.83392334, 0.07568359],
    "a": [0.99572754, 0.94790649, 0.53567505],
}


class EsmModel:
    """Exponential sinusoidal model"""

    def __init__(
        self,
        deltas: npt.NDArray[float],
        nus: npt.NDArray[float],
        amps: npt.NDArray[float],
        phis: npt.NDArray[float],
    ) -> None:
        self.deltas = np.array(deltas)
        self.nus = np.array(nus)
        self.amps = np.array(amps)
        self.phis = np.array(phis)
        s = self.deltas.shape
        assert (
            self.nus.shape == s and self.amps.shape == s and self.phis.shape == s
        ), "Inconsistent argument shape."
        self.r = s[0]

    def synth(
        self,
        n_s: int,
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
        log_zs = -self.deltas + 1j * 2 * np.pi * self.nus  # log of poles
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
