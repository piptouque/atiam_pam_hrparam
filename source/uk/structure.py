
import copy
import numpy as np
import numpy.typing as npt
import scipy.integrate as integrate

from typing import Callable, Union, Tuple, List
from abc import abstractmethod, abstractproperty

from uk.data import GuitarStringData, GuitarBodyData, Excitation, AFloat, AInt, ACallableFloat


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
    def __init__(self, nb_modes: int, nb_steps: int, h: float) -> None:
        self.n = np.arange(nb_modes)
        self.nb_steps = nb_steps
        self.h = h

        self._param_dict = {
            'n': self.n,
            'nb_steps': self.nb_steps,
            'h': h
        }

    @staticmethod
    def solve_constraints(structs: List[ModalStructure], a_ns: List[List[npt.NDArray[float]]], b_ns: List[npt.NDArray[float]]) -> List[List[npt.NDArray[float]]]:
        """
        For now, a_ns, b_ns are constants in time.
        a_ns[i] are the list of constraints applied on sub-structure i.
        b_ns[i] is the result of the constraints ?

        Args:
            a_ns (List[npt.NDArray[float]]): [description]
            b_ns (List[npt.NDArray[float]]): [description]
            n (AInt): [description]
        """
        assert len(a_ns) == len(structs) and \
            len(b_ns) == len(structs)
        #
        a_mat = np.array(a_ns)
        b_vec = np.array(b_ns)

        m_halfinv_mat = np.diag(chain.from_iterable(
            [np.pow(struct.m_n(self.n), -0.5) for struct in structs]))
        #
        b_mat = a_mat @ m_halfinv_mat
        b_plus_mat = np.linalg.pinv(b_mat)
        w_mat = 1 - m_halfinv_mat * b_plus_mat * a_mat
        return None

    def solve(self, structs: List[ModalStructure], ext_forces: List[Callable[[float, float], float]], q_n_is: List[AFloat], dq_n_is: List[AFloat]) -> Tuple[List[npt.NDArray[AFloat]]]:
        assert len(structs) == len(ext_forces) and len(
            structs) == len(q_n_is) and len(structs) == len(dq_n_is)

        def make_vec():
            if np.ndim(self.n) != 0:
                return [np.zeros(self.n.shape + (self.nb_steps,), dtype=float)
                        for i in range(len(structs))]
            else:
                return [np.zeros((self.nb_steps,), dtype=float) for i in range(len(structs))]
        q_ns = make_vec()
        dq_ns = make_vec()
        ddq_ns = make_vec()
        ddq_u_ns = make_vec()
        dq_half_ns = make_vec()
        ext_force_n_ts = make_vec()
        for i in range(len(structs)):
            q_ns[i][..., 0] = q_n_is[i]
            dq_ns[i][..., 0] = dq_n_is[i]
        #
        t = np.arange(self.nb_steps) * self.h
        for k in range(1, self.nb_steps):
            for i in range(len(structs)):
                struct = structs[i]
                q_ns[i][..., k] = q_ns[i][..., k-1] + self.h * dq_ns[i][..., k-1] + \
                    0.5 * self.h**2 * ddq_ns[i][..., k-1]
                dq_half_ns[i][..., k] = dq_ns[i][..., k-1] + \
                    0.5 * self.h * ddq_ns[i][..., k-1]
                ext_force_n = struct.ext_force_n(ext_forces[i], self.n)
                ext_force_n_ts[i][..., k] = ext_force_n(t[k])
                ddq_u_ns[i][..., k] = struct.solve_unconstrained(
                    q_ns[i][..., k], dq_half_ns[i][..., k], self.n, ext_force_n_ts[i][..., k])
                # solve constraints
                # TODO
                #
                ddq_ns[i][..., k] = ddq_u_ns[i][..., k]
                #
                dq_ns[i][..., k] = dq_ns[i][..., k-1] + 0.5 * \
                    self.h * (ddq_ns[i][..., k-1] + ddq_ns[i][..., k])
        return t, q_ns, dq_ns, ddq_ns, ext_force_n_ts
