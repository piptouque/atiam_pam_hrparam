
import copy
import numpy as np
import numpy.typing as npt
import scipy.integrate as integrate
from itertools import chain

from typing import Callable, Union, Tuple, List
from abc import abstractmethod, abstractproperty

import pandas as pd
import json
import argparse
from util import make_modetime_dataframe

AFloat = Union[float, npt.NDArray[float]]
AInt = Union[int, npt.NDArray[int]]

AVector = Union[Tuple[float, float], npt.NDArray[Tuple[float, float]]]

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
        raise NotImplemented

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
        raise NotImplemented

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

    @abstractmethod
    def phi_n(self, n: AInt) -> ACallableFloat:
        """Modeshapes

        Args:
            n (int): [description]

        Returns:
            Callable[[npt.NDArray[float]], npt.NDarray[float]]: The modeshape function for mode n
        """
        return NotImplemented

    @abstractproperty
    def extends(self) -> Tuple[float, float]:
        """Extends of the structure along the vibrating dimension

        Returns:
            Tuple[float, float]: [description]
        """
        return NotImplemented

    def ext_force_n(self, ext_force: Callable[[float, float], float], n: AInt) -> ACallableFloat:
        phi_n = self.phi_n(n)
        x_1, x_2 = self.extends

        def _f(t):
            if np.ndim(n) != 0:
                ext_f = np.empty(n.shape, dtype=float)
                for j in range(len(n)):
                    def _g(x, t):
                        f = ext_force(x, t)
                        phi = phi_n[j](x)
                        return f * phi
                    integ, err = integrate.quad(_g, x_1, x_2, args=(t))
                    ext_f[j] = integ
                return ext_f
            else:
                ext_f, err = integrate.quad(lambda x, t: ext_force(
                    x, t) * phi_n(x), x_1, x_2, args=(t))
                return ext_f
        return _f

    def solve_unconstrained(self, q_n: AFloat, dq_n: AFloat, n: AInt, ext_force_n_t: AFloat) -> AFloat:
        c_n = self.c_n(n)
        k_n = self.k_n(n)
        m_n = self.m_n(n)
        ddq_u_n = - (c_n * dq_n - k_n * q_n + ext_force_n_t) / m_n
        return ddq_u_n


class GuitarBody(ModalStructure):
    def __init__(self, data: GuitarBodyData) -> None:
        self.data = data

    def f_n(self, n: AInt) -> AFloat:
        return self.data.f_n[self.data.n[n]]

    def ksi_n(self, n: AInt) -> AFloat:
        return self.data.ksi_n[self.data.n[n]]

    def m_n(self, n: AInt) -> AFloat:
        return self.data.m_n[self.data.n[n]]

    def phi_n(self, n: AInt) -> AFloat:
        # no info on the modeshapes for the body.
        return np.full(n.shape, lambda x: 0, dtype=type(Callable)) if np.ndim(n) != 0 else lambda x: 0

    @ property
    def extends(self) -> Tuple[float, float]:
        # same thing, we don't know.
        return (0, 0)


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
        x_1, x_2 = self.extends
        if np.ndim(n) != 0:
            integs = np.empty(n.shape, dtype=float)
            for i in range(len(n)):
                integ, err = integrate.quad(
                    lambda x: phi_n[i](x)**2, x_1, x_2)
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

    def phi_n(self, n: AInt) -> ACallableFloat:
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

    @ property
    def extends(self) -> Tuple[float, float]:
        return (0, self.data.l)


class CouplingSolver:
    def __init__(self, structs: List[ModalStructure], ext_forces: List[Callable[[float, float], float]]) -> None:
        self.structs = structs
        self.ext_forces = ext_forces

    def solve_constraints(a_ns: List[List[npt.NDArray[float]]], b_ns: List[npt.NDArray[float]]) -> List[List[npt.NDArray[float]]]:
        """
        For now, a_ns, b_ns are constants in time.
        a_ns[i] are the list of constraints applied on sub-structure i.
        b_ns[i] is the result of the constraints ?

        Args:
            a_ns (List[npt.NDArray[float]]): [description]
            b_ns (List[npt.NDArray[float]]): [description]
            n (AInt): [description]
        """
        assert len(a_ns) == len(self.structs) and \
            len(b_ns) == len(self.structs)
        #
        a_mat = np.array(a_ns)
        b_vec = np.array(b_ns)

        m_halfinv_mat = np.diag(chain.from_iterable(
            [np.pow(struct.m_n(n), -0.5) for struct in self.structs]))
        #
        b_mat = a_mat @ m_halfinv_mat
        b_plus_mat = np.linalg.pinv(b_mat)
        w_mat = 1 - m_halfinv_mat * b_plus_mat * a_mat
        return None

    def solve(self, q_n_is: List[AFloat], dq_n_is: List[AFloat], n: AInt, nb_steps: int, h: float) -> Tuple[List[npt.NDArray[AFloat]]]:
        def make_vec():
            if np.ndim(n) != 0:
                return [np.zeros(n.shape + (nb_steps,), dtype=float)
                        for i in range(len(self.structs))]
            else:
                return [np.zeros((nb_steps,), dtype=float) for i in range(len(self.structs))]
        q_ns = make_vec()
        dq_ns = make_vec()
        ddq_ns = make_vec()
        ddq_u_ns = make_vec()
        dq_half_ns = make_vec()
        ext_force_n_ts = make_vec()
        for i in range(len(self.structs)):
            q_ns[i][..., 0] = q_n_is[i]
            dq_ns[i][..., 0] = dq_n_is[i]
        #
        t = np.arange(nb_steps) * h
        for k in range(1, nb_steps):
            for i in range(len(self.structs)):
                struct = self.structs[i]
                q_ns[i][..., k] = q_ns[i][..., k-1] + h * dq_ns[i][..., k-1] + \
                    0.5 * h**2 * ddq_ns[i][..., k-1]
                dq_half_ns[i][..., k] = dq_ns[i][..., k-1] + \
                    0.5 * h * ddq_ns[i][..., k-1]
                ext_force_n = struct.ext_force_n(self.ext_forces[i], n)
                ext_force_n_ts[i][..., k] = ext_force_n(t[k])
                ddq_u_ns[i][..., k] = struct.solve_unconstrained(
                    q_ns[i][..., k], dq_half_ns[i][..., k], n, ext_force_n_ts[i][..., k])
                # solve constraints
                # TODO
                #
                ddq_ns[i][..., k] = ddq_u_ns[i][..., k]
                #
                dq_ns[i][..., k] = dq_ns[i][..., k-1] + 0.5 * \
                    h * (ddq_ns[i][..., k-1] + ddq_ns[i][..., k])
        return t, q_ns, dq_ns, ddq_ns, ext_force_n_ts


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


def main(string_data: GuitarStringData, body_data: GuitarBodyData, f_ext_string: Callable[[float, float], float]):
    b = GuitarBody(body_data)
    s = GuitarString(string_data)

    f_ext_body = Excitation.make_null()

    solver = CouplingSolver([s, b], [f_ext_string, f_ext_body])

    n = np.arange(5)
    nb_steps = 8
    h = 0.01

    q_n_is = [np.zeros(n.shape, dtype=float) for i in range(2)]
    dq_n_is = [np.zeros(n.shape, dtype=float) for i in range(2)]

    t, q_ns, dq_ns, ddq_ns, ext_force_n_ts = solver.solve(
        q_n_is, dq_n_is, n, nb_steps, h)

    # data_q_n = cartesian_product(*q_ns[0])
    df_q_n = make_modetime_dataframe(q_ns[0], n, t)
    df_dq_n = make_modetime_dataframe(dq_ns[0], n, t)
    df_ddq_n = make_modetime_dataframe(ddq_ns[0], n, t)
    df_ext_force_n_t = make_modetime_dataframe(ext_force_n_ts[0], n, t)

    print(df_ddq_n)

    return df_q_n, df_dq_n, df_ddq_n, df_ext_force_n_t


def test():
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Guitar test')
    parser.add_argument('--string', required=True, type=str,
                        help='Guitar string config file path')
    parser.add_argument('--body', required=True, type=str,
                        help='Guitar body data file path')
    parser.add_argument('--excitation', required=True, type=str,
                        help='String excitation config file path')
    args = parser.parse_args()

    with open(args.string, mode='r') as config:
        string_data = json.load(
            config, object_hook=lambda d: GuitarStringData(**d))
    body_frame = pd.read_csv(args.body)
    body_frame = body_frame.to_dict()
    for (k, v) in body_frame.items():
        body_frame[k] = np.fromiter(v.values(), dtype=float)
    body_data = GuitarBodyData(**body_frame)
    with open(args.excitation, mode='r') as config:
        f_ext_string = json.load(
            config, object_hook=lambda d: Excitation.make_triangular(l=string_data.l, **d))
    main(string_data, body_data, f_ext_string)
