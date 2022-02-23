from typing import List, Tuple

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
        gammas = np.asarray(gammas)
        nus = np.asarray(nus)
        amps = np.asarray(amps)
        # Wrapping phases
        # see: https://stackoverflow.com/a/15927914
        phis = np.asarray(phis)
        phis = (phis + np.pi) % (2 * np.pi) - np.pi
        r = gammas.shape[0]
        # Some safety checks
        s = gammas.shape
        assert (
            gammas.ndim == 1 and nus.shape == s and amps.shape == s and phis.shape == s
        ), "Inconsistent argument shape."
        # assert np.all(gammas >= 0), "Dampings should be positive or null."
        assert np.all(-0.5 <= nus) and np.all(
            nus <= 0.5
        ), "Frequencies are normalised (reduced)."
        # sort by increasing frequencies
        incr_ids = np.argsort(nus, axis=-1)
        gammas = np.take_along_axis(gammas, incr_ids, axis=-1)
        nus = np.take_along_axis(nus, incr_ids, axis=-1)
        amps = np.take_along_axis(amps, incr_ids, axis=-1)
        phis = np.take_along_axis(phis, incr_ids, axis=-1)
        self.gammas = gammas
        self.nus = nus
        self.amps = amps
        self.phis = phis
        self.r = r

    @property
    def alphas(self) -> npt.NDArray[complex]:
        """_summary_

        Returns:
            npt.NDArray[complex]: _description_
        """
        return self.ampphase_to_alphas(self.amps, self.phis)

    @property
    def zs(self) -> npt.NDArray[complex]:
        """_summary_

        Returns:
            npt.NDArray[complex]: _description_
        """
        return self.dampfreq_to_poles(self.gammas, self.nus)

    @classmethod
    def from_complex(
        cls,
        zs: npt.NDArray[complex],
        alphas: npt.NDArray[complex],
        clip_damp: bool = False,
        discard_freq: bool = False,
    ) -> object:
        """Makes an ESM model given complex poles and comples amplitudes

        Args:
            zs (npt.NDArray[complex]): [description]
            alphas (npt.NDArray[complex]): [description]
            clip_damp (bool, optional): Floors the normalised damping factors to non-negative values. Defaults to False.
            discard_freq (bool, optional): Clips the normalised frequencies in the [-0.5, 0.5] range. Defaults to False.

        Returns:
            EsmModel: [description]
        """
        gammas, nus = cls.poles_to_dampfreq(
            zs, clip_damp=clip_damp, discard_freq=discard_freq
        )
        amps, phis = cls.alphas_to_ampphase(alphas)
        return cls(gammas, nus, amps, phis)

    @classmethod
    def poles_to_dampfreq(
        cls,
        zs: npt.NDArray[complex],
        clip_damp: bool = False,
        discard_freq: bool = False,
    ) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
        """Computes the normalised damping factors and frequencies from the complex poles.

        Estimated damping could be negative.
        In case of the non-adaptive ESPRIT,
        this is probably not wanted, so set the clip_damp argument to True
        In case of the adaptive ESPRIT,
        Negative dampings should be allowed,
        because the signal should be allowed to get stronger locally.

        Args:
            zs (npt.NDArray[complex]): [description]
            clip_damp (bool, optional): Floors the normalised damping factors to non-negative values. Defaults to False.
            discard_freq (bool, optional): Clips the normalised frequencies in the [-0.5, 0.5] range. Defaults to False.


        Returns:
            Tuple[npt.NDArray[float], npt.NDArray[float]]: [description]
        """
        # DAMPING RATIOS AND FREQUENCIES
        # damping factors
        gammas = -np.log(np.abs(zs))
        nus = np.angle(zs) / (2 * np.pi)  # frequencies
        gammas, nus = cls._fix_dampfreq(
            gammas, nus, clip_damp=clip_damp, discard_freq=discard_freq
        )
        return gammas, nus

    @classmethod
    def dampfreq_to_poles(
        cls, gammas: npt.NDArray[float], nus: npt.NDArray[float]
    ) -> npt.NDArray[complex]:
        """_summary_

        Args:
            gammas (npt.NDArray[float]): _description_
            nus (npt.NDArray[float]): _description_

        Returns:
            npt.NDArray[complex]: _description_
        """
        return np.exp(-gammas + 2j * np.pi * nus)

    @classmethod
    def fix_poles(
        cls, zs: npt.NDArray[complex], clip_damp: bool = True, discard_freq: bool = True
    ) -> npt.NDArray[complex]:
        """Corrects the poles to get appropriate damping factors (which should be in positive),
        and frequencies (which should be in [-0.5, 0.5])

        Args:
            zs (npt.NDArray[complex]): Complex poles
            clip_damp (bool, optional): Clips damping factors to [0, +inf[. Defaults to False.
            discard_freq (bool, optional): Discard poles whose estimated frequency is not in [-0.5, 0.5]. Defaults to False.

        Returns:
            npt.NDArray[complex]: 'Fixed' complex poles
        """
        gammas, nus = cls.poles_to_dampfreq(zs)
        gammas, nus = cls._fix_dampfreq(
            gammas, nus, clip_damp=clip_damp, discard_freq=discard_freq
        )
        zs = cls.dampfreq_to_poles(gammas, nus)
        return zs

    @staticmethod
    def _fix_dampfreq(
        gammas: npt.NDArray[float],
        nus: npt.NDArray[float],
        clip_damp: bool = False,
        discard_freq: bool = False,
    ) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
        """Corrects the damping factors and frequencies to get appropriate damping factors (which should be in positive),
        and frequencies (which should be in [-0.5, 0.5])

        Args:
            gammas (npt.NDArray[float]): Normalised damping factors
            nus (npt.NDArray[float]): Normalised frequencies
            clip_damp (bool, optional): Clips damping factors to [0, +inf[. Defaults to False.
            discard_freq (bool, optional): Discard poles whose estimated frequency is not in [-0.5, 0.5]. Defaults to False.

        Returns:

        Returns:
            Tuple[npt.NDArray[float], npt.NDArray[float]]: ('Fixed' normalised damping factors, 'Fixed' normalised frequencies)
        """
        if clip_damp:
            gammas = np.maximum(0, gammas)
        if discard_freq:
            # Normalised frequencies should
            # be in the [-0.5, 0.5] range.
            # Discard both frequency and damping if that's not the case
            ids_good = np.logical_and(-0.5 <= nus, nus <= 0.5)
            if not np.all(ids_good):
                print(nus)
            gammas = gammas[ids_good]
            nus = nus[ids_good]
            if not np.all(ids_good):
                print(nus)
        return gammas, nus

    @classmethod
    def alphas_to_ampphase(
        cls,
        alphas: npt.NDArray[complex],
    ) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
        """The normalised frequencies and normalised damping factors,

        Args:
            alphas (npt.NDArray[complex]): Complex amplitudes

        Returns:
            Tuple[npt.NDArray[float], npt.NDArray[float]]: (real amplitudes, initial phases)
        """
        amps = np.abs(alphas)
        phis = np.angle(alphas)
        return amps, phis

    @classmethod
    def ampphase_to_alphas(
        cls, amps: npt.NDArray[float], phis: npt.NDArray[float]
    ) -> npt.NDArray[complex]:
        """_summary_

        Args:
            amps (npt.NDArray[float]): _description_
            phis (npt.NDArray[float]): _description_

        Returns:
            npt.NDArray[complex]: _description_
        """
        return amps * np.exp(1j * phis)

    def synth(
        self,
        n_s: int,
    ) -> npt.NDArray[float]:
        """Synthesizes a sinusoidal signal, with set frequencies,
        amplitudes, initial phases,
        and damping factors.

        The frequencies $\nu_k$ and damping factors $\gamma_k$ should
        be those of a signal sampled at rate 1. We will call them 'normalised'.

        Args:
            n_s (int): Number of samples to synthesise

        Returns:
            npt.NDArray[float]: Synthesised temporal signal
        """
        ts = np.arange(n_s)  # discrete times
        log_zs = -self.gammas + 1j * 2 * np.pi * self.nus  # log of poles
        alphas = self.amps * np.exp(1j * self.phis)  # complex amplitudes

        # noisless signal
        x_synth = np.sum(
            np.outer(alphas, np.ones_like(ts)) * np.exp(np.outer(log_zs, ts)), axis=0
        )
        # x_synth = np.real(x_synth)
        return x_synth

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


class BlockEsmModel:
    """
    Adaptive/Block ESM model
    """

    def __init__(self, esm_list: List[EsmModel]) -> None:
        """Init from a list of ESM models

        Args:
            esm_list (List[EsmModel]): [description]
        """
        self.esm_blocks = esm_list

    @classmethod
    def from_param_lists(
        cls,
        gammas_list: npt.NDArray[float],
        nus_list: npt.NDArray[float],
        amps_list: npt.NDArray[float],
        phis_list: npt.NDArray[float],
    ) -> object:
        """[summary]

        Args:
            gammas (npt.NDArray[float]): (nb_blocks, r) shape. Normalised damping (multiply by sampling rate to get 'delta' in Amp.s-1)
            nus (npt.NDArray[float]): (nb_blocks, r) shape. Normalised frequencies, in [-0.5, 0.5]
            amps (npt.NDArray[float]): (nb_blocks, r) shape.
            phis (npt.NDArray[float]): (nb_blocks, r) shape.
        """
        s = np.shape(gammas_list)
        assert len(s) == 2
        nb_blocks = s[0]
        esm_list = []
        for j in range(nb_blocks):
            gammas = gammas_list[j]
            nus = nus_list[j]
            amps = amps_list[j]
            phis = phis_list[j]
            esm = EsmModel(gammas, nus, amps, phis)
            esm_list.append(esm)
        return cls(esm_list)

    def synth(
        self,
        n_s_block: int,
    ) -> npt.NDArray[float]:
        """[summary]

        Args:
            n_s_block (int): Number of samples for each block

        Returns:
            npt.NDArray[float]: Synthesised temporal signal
        """
        # no overlapping
        x_synth = np.empty(n_s_block * len(self.esm_blocks), dtype=complex)
        for j, esm_block in enumerate(self.esm_blocks):
            x_synth[j * n_s_block : (j + 1) * n_s_block] = esm_block.synth(n_s_block)
        return x_synth

    @property
    def nus(self) -> npt.NDArray[float]:
        """[summary]

        Returns:
            List[npt.NDArray[float]]: [description]
        """
        return np.stack([esm.nus for esm in self.esm_blocks], axis=1)

    @property
    def gammas(self) -> npt.NDArray[float]:
        """[summary]

        Returns:
            List[npt.NDArray[float]]: [description]
        """
        return np.stack([esm.gammas for esm in self.esm_blocks], axis=1)

    @property
    def amps(self) -> npt.NDArray[float]:
        """[summary]

        Returns:
            List[npt.NDArray[float]]: [description]
        """
        return np.stack([esm.amps for esm in self.esm_blocks], axis=1)

    @property
    def phis(self) -> npt.NDArray[float]:
        """[summary]

        Returns:
            List[npt.NDArray[float]]: [description]
        """
        return np.stack([esm.phis for esm in self.esm_blocks], axis=1)
