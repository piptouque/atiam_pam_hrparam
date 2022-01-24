import numpy as np
import numpy.typing as npt

from typing import Callable, Union

AFloat = Union[float, npt.NDArray[float]]
AInt = Union[int, npt.NDArray[int]]

ACallableFloat = Union[Callable[[AFloat], AFloat],
                       npt.NDArray[Callable[[AFloat], AFloat]]]


class GuitarStringData:
    def __init__(self, l: float, t: float, rho: float, e: float, i: float, eta_f: float, eta_a: float, eta_b: float) -> None:
        """
            Notations and values taken from:

            ANTUNES, Jose et DEBUT, Vincent.
            Dynamical computation of constrained flexible systems
                using a modal Udwadia-Kalaba formulation: Application to musical instruments.
            The Journal of the Acoustical Society of America, 2017, vol. 141, no 2, p. 764-778.


        Args:
            l (float): length (m)
            t (float): tension (N)
            rho (float): mass per unit length (kg/m)
            e (float):
            i (float): inertia torque
            eta_f (float): 'internal friction' (no dim)
            eta_a (float): 'air viscous damping' (no dim)
            eta_b (float):  'bending damping' (no dim)
        """
        self.l = l
        self.t = t
        self.rho = rho
        # some values chosen at random for now
        self.e = e
        self.i = i

        # damping
        self.eta_f = eta_f
        self.eta_a = eta_a
        self.eta_b = eta_b

        self._param_dict = {
            'l': self.l,
            't': self.t,
            'rho': self.rho,
            'e': self.e,
            'i': self.i,
            'eta_f': self.eta_f,
            'eta_a': self.eta_a,
            'eta_b': self.eta_b,
            'c_t': self.c_t,
            'b': self.b
        }

    @ property
    def c_t(self) -> float:
        """Transverse wave propagation velocity (m/s)

        Returns:
            float: [description]
        """
        return np.sqrt(self.t / self.rho)

    @ property
    def b(self) -> float:
        """Inharmonicity parameter (bending stiffness of a non-ideal string)
        """
        return self.e * self.i


class GuitarBodyData:
    def __init__(self, n: AInt, f_n: AFloat, ksi_n: AFloat, m_n: AFloat) -> None:
        self.n = n
        self.f_n = f_n
        self.ksi_n = ksi_n
        self.m_n = m_n

        if np.ndim(n) != 0:
            assert (len(n) == len(f_n)) and (len(n) == len(
                ksi_n)) and (len(n) == len(m_n)), "Noooo"
            self.n = np.array(self.n, dtype=int)
            self.f_n = np.array(self.f_n)
            self.ksi_n = np.array(self.ksi_n)
            self.m_n = np.array(self.m_n)

        self._param_dict = {
            'n': self.n,
            'f_n': self.f_n,
            'ksi_n': self.ksi_n,
            'm_n': self.m_n
        }


class Excitation:
    @staticmethod
    def make_triangular(x_rel: float, l: float, height: float, delta_t: float) -> Callable[[float, float], float]:
        """Make a triangular function corresponding to
        a time-limited near-point excitation of a string.

        Args:
            x_rel (float): x-coordinate ratio of the position of the excitation.
            l (float): [description]
            height (float): [description]
            delta_t (float): [description]

        Returns:
            Callable[[float, float], float]: [description]
        """
        x_e = x_rel * l

        def triag(x: float, t: float) -> float:
            if t > delta_t:
                return 0
            elif 0 <= x < x_e:
                return height * x / x_e
            elif x_e <= x <= l:
                return height * (1 - (x - x_e) / (l - x_e))
            else:
                return 0
        return triag

    @staticmethod
    def make_null() -> Callable[[float, float], float]:
        return lambda x, t: 0
