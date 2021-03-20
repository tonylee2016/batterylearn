from abc import abstractmethod, ABC

import numpy as np


class Dynamical:

    def __init__(self, dimension: int = None, x0=None):
        if x0:
            self.x = x0
            self.dim = len(x0)
        elif dimension and dimension > 0:
            self.x = np.zeros((dimension,))
            self.dim = dimension

    @abstractmethod
    def ode(self, **kwargs):
        pass

    @abstractmethod
    def out(self, **kwargs):
        pass

    def update(self, x):
        self.x = x


class LinearTimeInvariant(Dynamical):
    def __init__(self, A, B, C=None, D=None, x0=None):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        super(LinearTimeInvariant, self).__init__(len(A), x0)

    def ode(self, **kwargs):
        dx = - np.linalg.inv(self.A).dot(self.B.dot(self.x))

        return dx

    def out(self, **kwargs):
        dx = self.C.dot(self.x)

        return dx


class NonlinearSystem:
    # todo: implement basic nonlinear system
    pass


class Mechanical(Dynamical):
    def __init__(self, K, C, M, x0=None):

        if K.shape != C.shape or K.shape != M.shape:
            ValueError('the input K-C-M parameters should have the same dimension')
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

    def __init__(self, K, C, M, x0=None):
        if K.shape != C.shape or K.shape != M.shape:
            ValueError('the input K-C-M parameters should have the same dimension')
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


class Magnetic:
    """ the basic class for magnetic element.
           Parameters
           ----------
           mu : relative permeability
            C : numpy array
               damping matrix, must be 5x5.
               C[1,0]: I_p: primary moment of inertia
               C[0,1]: I_p: primary moment of inertia
           M : numpy array
               inertial matrix, must be 5x5.
               M[0,0]: I_t: transversal moment of inertia
               M[1,1]: I_t: transversal moment of inertia
               M[2,2]: m: flywheel mass
               M[3,3]: m: flywheel mass
               M[4,4]: m: flywheel mass
           x0 : numpy array (default: None)
                intial states must be 1x5.
           Attributes
           ----------
           x : numpy.ndarray.
               internal states, Nx1.
               x[0]:\theta_x
               x[1]:\theta_y
               x[2]:x
               x[3]:y
               x[4]:z
           Notes
           -----
               wip.
           Examples
           --------
           >>> wip
            References
            ----------
            ..  Li, Xiaojun & Palazzolo, Alan. Multi‐Input‐Multi‐Output Control of a Utility‐Scale,
             Shaftless Energy Storage Flywheel with a 5‐DOF Combination Magnetic Bearing.
             Journal of Dynamic Systems, Measurement, and Control 2018.
        """
