
import copy
import numpy as np
import numpy.typing as npt
import scipy.integrate as integrate

from typing import Callable, Union


class GuitarData:
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

    @property
    def c(self) -> float:
        """Transverse wave propagation velocity (m/s)

        Returns:
            float: [description]
        """
        return np.sqrt(self.t / self.rho)

    @property
    def b(self) -> float:
        """Inharmonicity parameter (bending stiffness of a non-ideal string)
        """
        return self.e * self.i


class GuitarString:
    def __init__(self,  data: GuitarData) -> None:
        """
            One difference: the indices start at 0,
            so f_0 is the fundamental.

        """
        self.data = data

    def _p_n(self, n: Union[int, npt.NDArray[int]]) -> Union[float, npt.NDArray[float]]:
        """Some modal factor

        Args:
            n (int): [description]

        Returns:
            float: [description]
        """
        return (2*n + 1) * np.pi / (2 * self.data.l)

    def f_n(self, n: Union[int, npt.NDArray[int]]) -> Union[float, npt.NDArray[float]]:
        """Modal frequencies

        Args:
            n (int): [description]

        Returns:
            float: [description]
        """
        p_n = self._p_n(n)
        return self.data.c / (2 * np.pi) * p_n * (1 + p_n ** 2 * self.data.b / (2 * self.data.t))

    def m_n(self, n: Union[int, npt.NDArray[int]]) -> Union[float, npt.NDArray[float]]:
        """Modal masses

        Args:
            n (int): [description]

        Returns:
            float: [description]
        """
        phi_n = self.phi_n(n)
        if np.ndim(n) != 0:
            integs = np.empty(n.shape, dtype=float)
            for i in range(len(n)):
                integ, err = integrate.quad(
                    lambda x: phi_n[i](x)**2, 0, self.data.l)
                integs[i] = integ
            return self.data.rho * integs
        else:
            integ, err = integrate.quad(lambda x:  phi_n(x)**2, 0, self.data.l)
            return self.data.rho * integ

    def ksi_n(self, n: Union[int, npt.NDArray[int]]) -> Union[float, npt.NDArray[float]]:
        p_n = self._p_n(n)
        f_n = self.f_n(n)
        return (self.data.t * (self.data.eta_f + self.data.eta_a / (2 * np.pi * f_n))
                + self.data.eta_b * self.data.b * p_n ** 2) / (2 * (self.data.t + self.data.b * p_n**2))

    def phi_n(self, n: Union[npt.NDArray[int]]) -> Union[Callable[[Union[float, npt.NDArray[float]]], Union[float, npt.NDArray[float]]], npt.NDArray[Callable[[Union[float, npt.NDArray[float]]], Union[float, npt.NDArray[float]]]]]:
        """Modeshapes

        Args:
            n (int): [description]

        Returns:
            Callable[[npt.NDArray[float]], npt.NDarray[float]]: The modeshape function for mode n
        """
        p_n = self._p_n(n)
        if np.ndim(n) != 0:
            phi_ns = np.empty(n.shape, dtype=type(Callable))
            for i in range(len(n)):

                # fix: copy value p_n[i] otherwise it is taken by reference.
                # defining function inside loops is tricky
                # see: https://stackoverflow.com/a/44385625
                def phi(x, p=p_n[i]):
                    return np.sin(p * x)
                phi_ns[i] = phi
            return phi_ns
        else:
            return lambda x: np.sin(p_n * x)


if __name__ == "__main__":
    import pandas as pd
    import json
    import argparse

    parser = argparse.ArgumentParser(description='Guitar test')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    args = parser.parse_args()

    with open(args.config, mode='r') as config:
        data = json.load(config, object_hook=lambda d: GuitarData(**d))
    s = GuitarString(data)

    n = np.arange(10)
    x = np.linspace(0, s.data.l, 50)
    v = np.empty((len(n), len(x)))
    phi = s.phi_n(n)
    for i in range(len(n)):
        v[i] = phi[i](x)
    d = pd.DataFrame({
        'f_n': s.f_n(n),
        'ksi_n': s.ksi_n(n),
        'm_n': s.m_n(n),
    }, index=list(n))
    d_p = pd.DataFrame(v)

    print(d)
    print(d_p)
