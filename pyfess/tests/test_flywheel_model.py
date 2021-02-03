from pyfess.models import flywheel
import numpy as np
import scipy as sp


def test_flywheel_model_with_bearing():
    K = np.diag([5, 5, 5, 5, 5])  # add some
    C = np.zeros(5, 5)
    C[0, 1] = 100
    C[1, 0] = -100
    M = np.diag([10, 10, 5, 5, 5])
    x0 = np.arange([0.1, 0.1, 0.1, 0.1, 0.1])

    fw = flywheel(K, C, M, x0=x0, w0=500)
