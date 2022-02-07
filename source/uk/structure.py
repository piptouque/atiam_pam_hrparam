from itertools import chain
import copy
import numpy as np
import numpy.typing as npt
import scipy.integrate as integrate

from typing import Callable, Union, Tuple, List
from abc import abstractmethod, abstractproperty

from uk.data import GuitarStringData, GuitarBodyData, AFloat, AInt, ACallableFloat, ACallableFloatVec


class Force:
    """[summary]
    """

    @abstractmethod
    def project_mode(self, phi: Callable[[float], float]) -> ACallableFloat:
        return NotImplemented

    @abstractmethod
    def __call__(self, x: AFloat, y: AFloat) -> AFloat:
        return NotImplemented


class ForceNull(Force):
    """Placeholder constant null function.
    """

    def project_mode(self, phi: Callable[[float], float]) -> ACallableFloat:
        return lambda t: 0

    def __call__(self, x: AFloat, y: AFloat) -> AFloat:
        return np.zeros_like(x) if np.ndim(x) != 0 else 0


class ForceRamp(Force):
    """[summary]
    """

    def __init__(self, x_rel: float, l: float, height: float, delta_x: float, delta_t: float) -> None:
        self.x_rel = x_rel
        self.l = l
        self.height = height
        self.delta_x = delta_x
        self.delta_t = delta_t

    def project_mode(self, phi: Callable[[float], float]) -> ACallableFloat:
        def _force_n(t: AFloat) -> AFloat:
            # if the force is a Dirac in space, simplify the integral instead of computing it
            # in order to prevent numerical errors.
            if not np.isclose(self.delta_x, 0):
                phi_val, err = integrate.quad(
                    phi, (self.x_rel - self.delta_x/2)*self.l, (self.x_rel + self.delta_x/2)*self.l)
            else:
                phi_val = phi(self.x_rel * self.l)
            return (t <= self.delta_t) * self.height * t / self.delta_t * phi_val
        return _force_n

    def __call__(self, x: AFloat, t: AFloat) -> AFloat:
        val = (t <= self.delta_t) * \
            (np.abs(x/self.l - self.x_rel) <= self.delta_x/2)
        return self.height * t / self.delta_t * val


class ModalStructure:
    """A base trait for modal structures.
        The modes start at 0,
        so f_0 is the fundamental.

    """
    @abstractmethod
    def f_n(self, n: AInt) -> AFloat:
        """Modal frequencies

        Args:
            n (AInt): The mode as an integer or modes as an array of integers.

        Returns:
            AFloat: The modal frequency ratio for mode `n` if `n` is an integer, an array of modal frequencies if `n` is an array.
        """
        raise NotImplemented

    @abstractmethod
    def ksi_n(self, n: AInt) -> AFloat:
        """Modal damping ratio

        Args:
            n (AInt): The mode as an integer or modes as an array of integers.

        Returns:
            AFloat: The modal damping ratio for mode `n` if `n` is an integer, an array of modal damping ratios if `n` is an array.
        """
        raise NotImplemented

    @abstractmethod
    def m_n(self, n: AInt) -> AFloat:
        """Modal masses

        Args:
            n (AInt): The mode as an integer or modes as an array of integers.

        Returns:
            AFloat: The modal mass for mode `n` if `n` is an integer, an array of modal masses if `n` is an array.
        """
        raise NotImplemented

    def c_n(self, n: AInt) -> AFloat:
        """Modal 'resistance'

        Args:
            n (AInt): The mode as an integer or modes as an array of integers.

        Returns:
            AFloat: The modal resistance for mode `n` if `n` is an integer, an array of modal resistances if `n` is an array.
        """
        return (4 * np.pi) * self.m_n(n) * self.f_n(n) * self.ksi_n(n)

    def k_n(self, n: AInt) -> AFloat:
        """Modal stiffness

        Args:
            n (AInt): The mode as an integer or modes as an array of integers.

        Returns:
            AFloat: The modal stiffness for mode `n` if `n` is an integer, an array of modal stiffness if `n` is an array.
        """
        return (4 * np.pi**2) * self.m_n(n) * self.f_n(n) ** 2

    @abstractmethod
    def phi_n(self, n: AInt) -> ACallableFloat:
        """Modeshapes.

        Args:
            n (AInt): The mode as an integer or modes as an array of integers.

        Returns:
            ACallableFloat: The modeshape function for mode `n` if `n` is an integer, an array of modeshape functions if `n` is an array.
        """
        return NotImplemented

    @abstractmethod
    def finger_coupling(self, n: AInt, finger_rel_pos: float) -> AFloat:
        return NotImplemented

    @abstractmethod
    def bridge_coupling(self, n: AInt, case: str) -> AFloat:
        return NotImplemented

    @abstractproperty
    def extends(self) -> Tuple[float, float]:
        """Extends of the structure along the vibrating dimension.

        Returns:
            Tuple[float, float]: [description]
        """
        return NotImplemented

    def ext_force_n(self, ext_force: Force, n: AInt) -> ACallableFloat:
        """Make the modal external force functions from the external force.

        Args:
            ext_force (Callable[[float, float], float]): The (x, t) function of external forces applied on the structure.
            n (AInt): The mode as an integer or modes as an array of integers.

        Returns:
            ACallableFloat: A (t) function if `n` is an integer, an array of (t) functions if `n` is an array.
        """
        phi_n = self.phi_n(n)

        if np.ndim(n) != 0:
            force_n = np.empty(n.shape, dtype=type(Callable))
            for j in range(len(n)):
                force = ext_force.project_mode(phi_n[j])
                force_n[j] = force
            return force_n
        else:
            force_n = ext_force.project_mode(phi_n)
            return force_n

    def solve_unconstrained(self, q_n: AFloat, dq_n: AFloat, n: AInt, ext_force_n_t: AFloat) -> AFloat:
        """Solves the unconstrained system.
        See equation (42) in:

            ANTUNES, Jose et DEBUT, Vincent.
            Dynamical computation of constrained flexible systems
                using a modal Udwadia-Kalaba formulation: Application to musical instruments.
            The Journal of the Acoustical Society of America, 2017, vol. 141, no 2, p. 764-778.

        Args:
            q_n (AFloat): The modal responses,
            dq_n (AFloat): The derivate of the modal responses `q_n`s,
            n (AInt): The mode as an integer or modes as an array of integers.
            ext_force_n_t (AFloat): The modal external forces.

        Returns:
            AFloat: The double-derivate of the unconstrained modal responses.
        """
        c_n = self.c_n(n)
        k_n = self.k_n(n)
        m_n = self.m_n(n)
        ddq_u_n = (- c_n * dq_n - k_n * q_n + ext_force_n_t) / m_n
        return ddq_u_n

    def y_n(self, q_n: AFloat, n: AInt) -> ACallableFloatVec:
        """[summary]

        Args:
            q_n (AFloat): The modal responses, a 
            n (AInt): The mode as an integer or modes as an array of integers.

        Returns:
            ACallableFloatVec: For each mode, a (x) function 
                returning the displacement as a time array:
                    [y_n(t=0), ... y_n(t=t_f)]
        """
        phi_n = self.phi_n(n)
        if np.ndim(n) != 0:
            y_n = np.empty(n.shape, dtype=type(Callable))

            def _y_j(j: int):
                return lambda x: phi_n[j](x) * q_n[j]
            for j in range(len(n)):
                y_n[j] = _y_j(j)
            return y_n
        else:
            return lambda x: phi_n(x) * q_n


class GuitarBody(ModalStructure):
    """Model for the inert guitar body.
    """

    def __init__(self, data: GuitarBodyData) -> None:
        self.data = data

    def _find_n(self, ids: AInt) -> AInt:
        # print("In _find_n")
        # print(self.data.n)
        # print(ids)
        # print(np.ndim(ids))
        if np.ndim(ids) > 0:
            n = np.empty_like(ids)
            for (j, idx) in enumerate(ids):
                n_idx, = np.where(self.data.n == idx)
                n[j] = n_idx
            return n
        else:
            return np.where(self.data.n == ids)

    def f_n(self, n: AInt) -> AFloat:
        return self.data.f_n[self._find_n(n)]

    def ksi_n(self, n: AInt) -> AFloat:
        return self.data.ksi_n[self._find_n(n)]

    def m_n(self, n: AInt) -> AFloat:
        return self.data.m_n[self._find_n(n)]

    def phi_n(self, n: AInt) -> AFloat:
        # no info on the modeshapes for the body.
        # we enforce a constant value of the bridge modeshapes
        # FIX version multidimensionnelle ne marche pas
        return np.array([lambda x: self.data.phi_n[ids] for ids in n], dtype=type(Callable)) if np.ndim(n) != 0 else lambda x: self.data.phi_n[n]

    def bridge_coupling(self, n: AInt, case="rigid") -> AFloat:
        if case == "rigid":
            return np.zeros((1, len(n)), dtype=float)
        elif case == "flexible":
            a_mat = np.zeros((1, len(n)), dtype=float)
            for j in range(len(n)):
                a_mat[0, j] = -self.phi_n(j)(0)
            return a_mat

    def finger_coupling(self, n: AInt, finger_rel_pos: float) -> AFloat:
        """
        Should always be zero since there is no finger constraint on the body.
        """
        return np.zeros((1, len(n)), dtype=float)

    @ property
    def extends(self) -> Tuple[float, float]:
        # same thing, we don't know.
        return (0, 0)


class GuitarString(ModalStructure):
    """Model for the guitar string.
    """

    def __init__(self,  data: GuitarStringData) -> None:
        self.data = data

    def _p_n(self, n: AInt) -> AFloat:
        """Some modal factor

        Args:
            n (AInt): The mode as an integer or modes as an array of integers.

        Returns:
            AFloat: Modal factors used in some other modal parameters.
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
            def _phi(j: int):
                return lambda x: np.sin(p_n[j] * x)
            phi_ns = np.empty(n.shape, dtype=type(Callable))
            for j in range(len(n)):

                # fix: copy value p_n[i] otherwise it is taken by reference.
                # defining function inside loops is tricky
                # see: https://stackoverflow.com/a/44385625
                phi_ns[j] = _phi(j)
            return phi_ns
        else:
            return lambda x: np.sin(p_n * x)

    def finger_coupling(self, n: AInt, finger_rel_pos: float):
        """
        constraint matrix for a finger on the fretboard.

        Args:
            n (AInt): array of modes;
            finger_rel_pos (float): finger position on the fretboard
                relatively to the length of the string.
        Returns:
            a_mat (AFloat)
        """
        a_mat = np.zeros((1, len(n)), dtype=float)
        phi_n = self.phi_n(n)
        for j in range(len(n)):
            a_mat[0, j] = phi_n[j](self.data.l * finger_rel_pos)
        return a_mat

    def bridge_coupling(self, n: AInt, case='rigid'):
        """
        constraint matrix for a rigid body constraining the string at the bridge.
        """
        a_mat = np.zeros((1, len(n)), dtype=float)
        phi_n = self.phi_n(n)
        for j in range(len(n)):
            a_mat[0, j] = phi_n[j](self.data.l)
        return a_mat

    @ property
    def extends(self) -> Tuple[float, float]:
        return (0, self.data.l)


class ModalSimulation:
    """A constraint solver and simulation for the U-K formulation of modal analysis.
    """

    def __init__(self, nb_modes: (int or list), nb_steps: int, h: float, num_struct=None, coupling="rigid", finger=0) -> None:
        if isinstance(nb_modes, int):
            if num_struct is None:
                raise TypeError("number of structures num_struct should be defined if nb_modes is int")
            self.n = [np.arange(nb_modes)]*num_struct
        else:
            self.n = [np.arange(nb_mode) for nb_mode in nb_modes]
        self.nb_steps = nb_steps
        self.h = h
        self.coupling = coupling
        self.finger = finger

        self._param_dict = {
            'n': self.n,
            'nb_steps': self.nb_steps,
            'h': h,
            'coupling': self.coupling,
            'finger': self.finger
        }

    # @staticmethod
    def solve_constraints(self, structs: List[ModalStructure], a_ns: List[List[npt.NDArray[float]]]) -> List[List[npt.NDArray[float]]]:
        """
        For now, a_ns, b_ns are constants in time.
        a_ns[i] are the list of constraints applied on sub-structure i.
        b_ns[i] is the result of the constraints ?

        Args:
            a_ns (List[npt.NDArray[float]]): [description]
            b_ns (List[npt.NDArray[float]]): [description]
            n (AInt): [description]
        """
        assert len(a_ns) == len(structs)
        #
        a_mat = np.hstack(a_ns)
        #b_vec = np.array(b_ns)

        m_halfinv_mat = np.diag(list(chain.from_iterable(
            [np.power(struct.m_n(self.n[s]), -0.5) for s, struct in enumerate(structs)])))
        #
        b_mat = a_mat @ m_halfinv_mat
        b_plus_mat = np.linalg.pinv(b_mat)
        w_mat = np.eye(sum([len(self.n[s]) for s in range(len(self.n))])) - m_halfinv_mat @ b_plus_mat @ a_mat
        return w_mat

    def run(self, structs: List[ModalStructure], ext_forces: List[Callable[[float, float], float]], q_n_is: List[AFloat], dq_n_is: List[AFloat]) -> Tuple[List[npt.NDArray[AFloat]]]:
        """Solve the constrained system.
        Based on the velocity-Verlet algorithm described in:

            ANTUNES, Jose et DEBUT, Vincent.
            Dynamical computation of constrained flexible systems
                using a modal Udwadia-Kalaba formulation: Application to musical instruments.
            The Journal of the Acoustical Society of America, 2017, vol. 141, no 2, p. 764-778.

        Args:
            structs (List[ModalStructure]): List of modal structures.
            ext_forces (List[Callable[[float, float], float]]): List of external forces applied to each modal structure.
            q_n_is (List[AFloat]): List of initial modal responses for each modal structure.
            dq_n_is (List[AFloat]): List of initial derivative of modal responses for each modal structure.

        Returns:
            Tuple[List[npt.NDArray[AFloat]]]: Tuple of computed times, modal responses, associated derivative and modal external forces for each modal structure.
        """
        assert len(structs) == len(ext_forces) and len(
            structs) == len(q_n_is) and len(structs) == len(dq_n_is)

        finger_constraints = []
        for struct in structs:
            if isinstance(struct, GuitarString):
                finger_constraints.append(self.finger)
            elif isinstance(struct, GuitarBody):
                finger_constraints.append(0)

        def _make_vec():
            if np.ndim(self.n) != 0:
                return [np.zeros(self.n[i].shape + (self.nb_steps,), dtype=float)
                        for i in range(len(structs))]
            else:
                return [np.zeros((self.nb_steps,), dtype=float) for i in range(len(structs))]
        q_ns = _make_vec()
        dq_ns = _make_vec()
        ddq_ns = _make_vec()
        ddq_u_ns = _make_vec()
        dq_half_ns = _make_vec()
        ext_force_n_ts = _make_vec()
        a_ns = []
        for i in range(len(structs)):
            q_ns[i][..., 0] = q_n_is[i]
            dq_ns[i][..., 0] = dq_n_is[i]
            a_bridge = structs[i].bridge_coupling(self.n[i], self.coupling)
            a_finger = structs[i].finger_coupling(self.n[i], finger_constraints[i])
            a_ns.append(np.vstack((a_bridge, a_finger)))
            print(a_ns)
        #
        t = np.arange(self.nb_steps) * self.h
        w_mat = self.solve_constraints(structs, a_ns)
        w_mat_list = []
        idx = np.cumsum([len(num_mode) for num_mode in self.n])
        idx = np.insert(idx, 0, 0)
        for i in range(len(structs)):
            w_mat_list.append(w_mat[idx[i]:idx[i+1], idx[i]:idx[i+1]])
        for k in range(1, self.nb_steps):
            for i in range(len(structs)):
                struct = structs[i]
                q_ns[i][..., k] = q_ns[i][..., k-1] + self.h * dq_ns[i][..., k-1] + \
                    0.5 * self.h**2 * ddq_ns[i][..., k-1]
                dq_half_ns[i][..., k] = dq_ns[i][..., k-1] + \
                    0.5 * self.h * ddq_ns[i][..., k-1]
                ext_force_n = struct.ext_force_n(ext_forces[i], self.n[i])
                if np.ndim(self.n) != 0:
                    for j in range(len(self.n[i])):
                        ext_force_n_ts[i][j, k] = ext_force_n[j](t[k])
                else:
                    ext_force_n_ts[i][..., k] = ext_force_n(t[k])
                ddq_u_ns[i][..., k] = struct.solve_unconstrained(
                    q_ns[i][..., k], dq_half_ns[i][..., k], self.n[i], ext_force_n_ts[i][..., k])
                # solve constraints
                #
                ddq_ns[i][..., k] = w_mat_list[i] @ ddq_u_ns[i][..., k]
                #ddq_ns[i][..., k] = (w_mat @ np.vstack(ddq_u_ns)[..., k])[idx[i]:idx[i+1]]
                #
                dq_ns[i][..., k] = dq_ns[i][..., k-1] + 0.5 * \
                    self.h * (ddq_ns[i][..., k-1] + ddq_ns[i][..., k])
        return t, q_ns, dq_ns, ddq_ns, ext_force_n_ts
