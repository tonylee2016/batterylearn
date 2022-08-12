import logging
from abc import abstractmethod

import numpy as np

# from ahkab import circuit


class Base:
    def __init__(self, type, name=""):
        self.name = name
        self.type = type
        logging.info(self.name + " of [" + self.type + "] is created")

    def __del__(self):
        logging.info(self.name + " is deleted")

    def show(self):
        logging.info(self.name + ' of "' + self.type + '"')


class Container:
    def __init__(self):
        self.items = dict()

    def attach(self, item: Base):
        self.items[item.name] = item
        return self

    def get(self, name):
        return self.items[name]


class Dynamical:
    """class for modeling a dynamical system
    """
    def __init__(self, dimension: int = None, x0=None):
        if x0:
            self.x = x0
            self.dim = len(x0)
        elif dimension and dimension > 0:
            self.x = np.zeros((dimension,))
            self.dim = dimension

    @abstractmethod
    def ode(self, **kwargs):
        """the ordinary differential equation that describes system dynamics
        """
        pass

    @abstractmethod
    def out(self, **kwargs):
        pass

    def update(self, x):
        self.x = x


class LinearTimeInvariant(Dynamical):
    """linear system

    Args:
        A (np.matrix): _description_ 
        B (np.array): _description_
        C (np.array, optional): _description_. Defaults to None.
        D (np.array, optional): _description_. Defaults to None.
        x0 (np.array, optional): _description_. Defaults to None.
    """
    def __init__(self, A:np.matrix, B:np.array, C:np.array=None, D:np.array=None, x0:np.array=None):

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        super(LinearTimeInvariant, self).__init__(len(A), x0)

    def ode(self, **kwargs):
        dx = -np.linalg.inv(self.A).dot(self.B.dot(self.x))

        return dx

    def out(self, **kwargs):
        dx = self.C.dot(self.x)

        return dx


class NonlinearSystem:
    # todo: implement basic nonlinear system
    pass


class Mechanical(Dynamical):

    """The mechanical system, which uses KCM (genertic, nonlinear) parameters 

    Args:
        K (np.array): stiffness array
        C (np.array): damping array
        M (np.array): mass/inertia array
        x0 (np.array, optional): initial conditons. Defaults to None.
    """

    def __init__(self, K, C, M, x0=None):

        if K.shape != C.shape or K.shape != M.shape:
            ValueError(
                "the input K-C-M parameters should have the same dimension"
            )
        else:
            self.K = K
            self.C = C
            self.M = M

        super().__init__(dimension=2 * len(K), x0=x0)

    def ode(self, **kwargs):
        pass

    def out(self, **kwargs):
        pass


class LinearMechanical(LinearTimeInvariant):

    """The linear mechanical system, which uses KCM parameters 

    Args:
        K (np.array): stiffness array
        C (np.array): damping array
        M (np.array): mass/inertia array
        x0 (np.array, optional): initial conditons. Defaults to None.
    """

    def __init__(self, K:np.array, C:np.array, M:np.array, x0:np.array=None):
        if K.shape != C.shape or K.shape != M.shape:
            ValueError(
                "the input K-C-M parameters should have the same dimension"
            )
        else:
            self.K = K
            self.C = C
            self.M = M

            Eye = np.diag([1] * 5)
            A = np.zeros((10, 10))
            A[:5, :5] = self.C
            A[5:, :5] = self.M
            A[:5, 5:] = Eye

            B = np.zeros((10, 10))
            B[:5, :5] = self.K
            B[5:, 5:] = -Eye

        super().__init__(A=A, B=B)


# class Magnetic():
#     """the basic class for magnetic circuit.
#     Parameters
#     ----------
#     mu : relative permeability
#      C : numpy array
#         damping matrix, must be 5x5.
#         C[1,0]: I_p: primary moment of inertia
#         C[0,1]: I_p: primary moment of inertia
#     R : numpy array
#         inertial matrix, must be 5x5.
#     mmf : numpy array (default: None)
#          intial states must be 1x5.
#     Attributes
#     ----------
#     x : numpy.ndarray.
#     Notes
#     -----
#         wip.
#     Examples
#     --------
#     >>> wip
#      References
#      ----------
#      ..  A Combination 5-DOF Active Magnetic Bearing For Energy Storage Flywheel,
#          Xiaojun Li and Alan Palazzolo and Zhiyang Wang, 2021, arXiv:2103.08004.
#     """

#     def __init__(self, title):
#         super().__init__(title)

#     def add_pm(self, part_id, n1, n2, l_pm, h_c):
#         """Adds permanent source to the circuit (also takes care that the nodes
#         are added as well).

#         **Parameters:**

#         part_id : string
#             The voltage source part_id (eg "VA"). The first letter is always V.
#         n1, n2 : string
#             The nodes to which the element is connected. Eg. ``"in"`` or
#             ``"out_a"``.
#         dc_value : float
#             DC voltage value
#         ac_value : float, optional
#             AC voltage value, defaults to 0.
#         function : function, optional
#             Time function. See devices.py for built-in options.
#         """
#         # self._add_vsource(self, part_id, n1, n2, dc_value)
#         # self._add_resistor(self, part_id, n1, n2, dc_value)
#         pass
