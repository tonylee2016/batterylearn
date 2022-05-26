import matplotlib.pyplot as plt
import numpy as np

from pyens.elements import Base, Dynamical


class OCV(Base):
    def __init__(self, name, soc=list(), ocv=list()):
        Base.__init__(self, type="soc_ocv_curve", name=name)
        self.ocv = ocv
        self.soc = soc

    def display(self):
        plt.figure()
        plt.xlabel("soc")
        plt.ylabel("ocv")
        plt.title("curve")
        plt.plot(self.soc, self.ocv)
        plt.show()

    def soc2ocv(self, soc):
        return np.interp(soc, self.soc, self.ocv)


class EcmCell(Base, Dynamical):
    def __init__(self, name, parameters: dict):
        Base.__init__(self, type="EMC_Cell_Model", name=name)
        Dynamical.__init__(self)
        self.parameters = parameters

    def ode(self, t, x, current_series):

        current = -(current_series(t))

        # The derivative of SoC
        if current >= 0:
            dSoC = -1 / self.parameters["CAP"] * current * 1.0
        else:
            dSoC = (
                -1 / self.parameters["CAP"] * current * self.parameters["ce"]
            )
        # The derivative of U1
        du1 = (
            1
            / self.parameters["C1"]
            * (current - 1 / self.parameters["R1"] * x[1])
        )
        # The derivative of U2
        du2 = (
            1
            / self.parameters["C2"]
            * (current - 1 / self.parameters["R2"] * x[2])
        )

        return np.array([du1, du2, dSoC])

    def out(self, **kwargs):
        pass
