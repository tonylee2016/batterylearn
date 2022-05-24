import numpy as np

from pyfess.elements import Mechanical


class Flywheel(Mechanical):
    """flywheel model class.
       Parameters
       ----------
       K : numpy array
           stiffness matrix, must be 5x5. should be all zeros when supported by AMB
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

    def __init__(self,
                 K: np.ndarray,
                 C: np.ndarray,
                 M: np.ndarray,
                 x0=None,
                 w0=None):

        super().__init__(K=K, C=C, M=M, x0=x0)

        if not w0:
            self.w = 0
        elif isinstance(w0, float) and w0 > 0:
            self.w = w0
        else:
            raise ValueError('intial spinning speed must float and positive')

    def ode(self, t, x, **kwargs):

        if not kwargs:
            f = np.zeros((self.dim, 1))

        dx = np.zeros((10,))
        self.x = x

        dx[0] = self.x[5]
        dx[1] = self.x[6]
        dx[2] = self.x[7]
        dx[3] = self.x[8]
        dx[4] = self.x[9]

        dx[5] = 1 / self.M[0, 0] * (f[0] - self.C[1, 0] * self.w * self.x[6] - self.K[0, 0] * self.x[0])
        dx[6] = 1 / self.M[1, 1] * (f[1] - self.C[0, 1] * self.w * self.x[5] - self.K[1, 1] * self.x[1])
        dx[7] = 1 / self.M[2, 2] * (f[2] - self.K[2, 2] * self.x[2])
        dx[8] = 1 / self.M[3, 3] * (f[3] - self.K[3, 3] * self.x[3])
        dx[9] = 1 / self.M[4, 4] * (f[4] - self.K[4, 4] * self.x[4])

        return dx

    # def connect(self,obj):
    #     if isinstance(obj,magnetic_bearing):


class MagneticBearing:
    """flywheel model class.
       Parameters
       ----------
       K : numpy array
           stiffness matrix, must be 5x5. should be all zeros when supported by AMB
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
        ..  Li, Xiaojun (2018). Design and Development of a Next Generation Energy Storage Flywheel.
         Doctoral dissertation, Texas A&M University.
         Available electronically from https : / /hdl .handle .net /1969 .1 /188891.
    """
    # def __init__(self):
