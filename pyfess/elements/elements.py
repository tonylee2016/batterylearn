from abc import abstractmethod, ABC

import numpy as np


class dynamical:

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


class linear_time_invariant(dynamical):
    def __init__(self, A, B, C=None, D=None, x0=None):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        super(linear_time_invariant, self).__init__(len(A), x0)

    def ode(self, **kwargs):
        dx = - np.linalg.inv(self.A).dot(self.B.dot(self.x))

        return dx

    def out(self, **kwargs):
        dx = self.C.dot(self.x)

        return dx


class Nonlinear_sys:
    pass


# todo: implement basic nonlinear system

class mechanical(dynamical):
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


class lin_mechanical(linear_time_invariant):

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
