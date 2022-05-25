import matplotlib.pyplot as plt
import numpy as np
from pyess.models import Flywheel, OCV
from pyess.utilities import ivp


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
    fw = Flywheel(K, C, M, w0=50.0, name="fw1")

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


def test_ocv_curve():
    ocv = [3.3, 3.5, 3.55, 3.6, 3.65, 3.68, 3.70, 3.8, 3.95, 4.0, 4.1]
    soc = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    c1 = OCV(name='curve1', ocv=ocv, soc=soc)

    assert c1.ocv == ocv
    assert c1.soc == soc
    c1.display()
