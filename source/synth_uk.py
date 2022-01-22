
import copy
import numpy as np
import numpy.typing as npt
import scipy.integrate as integrate

from typing import Callable, Union, Tuple, List
from abc import abstractmethod

AFloat = Union[float, npt.NDArray[float]]
AInt = Union[int, npt.NDArray[int]]

AVector = Union[Tuple[float, float], npt.NDArray[Tuple[float, float]]]

ACallableFloat = Union[Callable[[AFloat], AFloat],
                       npt.NDArray[Callable[[AFloat], AFloat]]]
ACallableVector = Union[Callable[[AVector], AVector],
                        npt.NDArray[Callable[[AVector], AVector]]]


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


class ModalStructure:
    """
        One difference: the indices start at 0,
        so f_0 is the fundamental.

    """
    @abstractmethod
    def f_n(self, n: AInt) -> AFloat:
        """Modal frequencies

        Args:
            n (int): [description]

        Returns:
            float: [description]
        """
        raise NotImplementedError()

    @abstractmethod
    def ksi_n(self, n: AInt) -> AFloat:
        """Modal damping ratio

        Args:
            n (AInt): [description]

        Raises:
            NotImplemented: [description]

        Returns:
            AFloat: [description]
        """
        raise Not()

    @abstractmethod
    def m_n(self, n: AInt) -> AFloat:
        """Modal masses

        Args:
            n (int): [description]

        Returns:
            float: [description]
        """
        raise NotImplementedError()


class GuitarBody(ModalStructure):
    def __init__(self, data: GuitarBodyData) -> None:
        self.data = data

    def f_n(self, n: AInt) -> AFloat:
        return self.data.f_n[self.n[n]]

    def ksi_n(self, n: AInt) -> AFloat:
        return self.data.ksi_n[self.n[n]]

    def m_n(self, n: AInt) -> AFloat:
        return self.data.m_n[self.n[n]]


class GuitarString(ModalStructure):
    def __init__(self,  data: GuitarStringData) -> None:
        self.data = data

    def _p_n(self, n: AInt) -> AFloat:
        """Some modal factor

        Args:
            n (int): [description]

        Returns:
            float: [description]
        """
        return (2*n + 1) * np.pi / (2 * self.data.l)

    def f_n(self, n: AInt) -> AFloat:
        p_n = self._p_n(n)
        return self.data.c_t / (2 * np.pi) * p_n * (1 + p_n ** 2 * self.data.b / (2 * self.data.t))

    def m_n(self, n: AInt) -> AFloat:
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

    def ksi_n(self, n: AInt) -> AFloat:
        p_n = self._p_n(n)
        f_n = self.f_n(n)
        return (self.data.t * (self.data.eta_f + self.data.eta_a / (2 * np.pi * f_n))
                + self.data.eta_b * self.data.b * p_n ** 2) / (2 * (self.data.t + self.data.b * p_n**2))

    def c_n(self, n: AInt) -> AFloat:
        """Modal 'resistance'

        Args:
            n (Uniont[int, npt.NDArray[int]]): [description]

        Returns:
            Union[float, npt.NDArray[float]]: [description]
        """
        return (4 * np.pi) * self.m_n(n) * self.f_n(n) * self.ksi_n(n)

    def k_n(self, n: AInt) -> AFloat:
        """Modal stiffness

        Args:
            n (Union[int, npt.NDArray[int]]): [description]

        Returns:
            Union[float, npt.NDArray[float]]: [description]
        """
        return (4 * np.pi**2) * self.m_n(n) * self.f_n(n) ** 2

    def phi_n(self, n: AInt) -> ACallableFloat:
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


class CouplingSolver:
    def __init__(self, structs: List[ModalStructure]) -> None:
        self.structs = structs

    def force_n(self, n: AInt) -> ACallableVector:
        # no f_ext for now
        c_ns = self.c_n(n)
        k_ns = self.k_n(n)


if __name__ == "__main__":
    import pandas as pd
    import json
    import argparse

    parser = argparse.ArgumentParser(description='Guitar test')
    parser.add_argument('--string', required=True, type=str,
                        help='Guitar string config file path')
    parser.add_argument('--body', required=True, type=str,
                        help='Guitar body data file path')
    args = parser.parse_args()

    with open(args.string, mode='r') as config:
        string_data = json.load(
            config, object_hook=lambda d: GuitarStringData(**d))
    body_frame = pd.read_csv(args.body)
    body_data = body_frame.to_numpy()
    b = GuitarBody(**body_data)
    s = GuitarString(string_data)

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
