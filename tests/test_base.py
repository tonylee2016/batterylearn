import matplotlib.pyplot as plt
import numpy as np
from pyfess.models import Flywheel
from pyfess.utilities import ivp


def test_flywheel_base():
    K = np.diag([5, 5, 5, 5, 5])  # add some
    C = np.zeros((5, 5))
    C[0, 1] = 100
    C[1, 0] = -100
    C[0, 0] = 1
    C[1, 1] = 1
    C[2, 2] = 1
    C[3, 3] = 1
    C[4, 4] = 1
    M = np.diag([10, 10, 5, 5, 5])
    fw = Flywheel(K, C, M, w0=50.0)

    v_t = np.arange(0, 10, 0.001)
    x0 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0])

    sol = ivp(fw.ode, x0=x0, v_t=v_t)

    plt.plot(
        sol.t,
        np.transpose(
            sol.y[
                :2,
            ]
        ),
    )
    plt.legend(["thetax", "thetay", "x", "y", "z"])
    plt.show()
    plt.plot()

def test_flywheel_model_with_bearing():
    pass
