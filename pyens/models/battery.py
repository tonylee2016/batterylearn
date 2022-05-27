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
    def __init__(self, name, parameters: dict, curve: OCV):
        Base.__init__(self, type="EMC_Cell_Model", name=name)
        Dynamical.__init__(self)
        self.parameters = parameters
        self.ocv_curve = curve
        self.SOC_RANGE = [0.0, 100.0]

    def ode(self, t, x, current_series):

        current = current_series(t)

        # dSoC/dt

        if self.SOC_RANGE[0] <= x[2] <= self.SOC_RANGE[1]:
            if current >= 0:
                dSoC = -1 / self.parameters["CAP"] * current / 36
            else:
                dSoC = (
                    -1
                    / self.parameters["CAP"]
                    * current
                    * self.parameters["ce"]
                    / 36
                )
        else:
            dSoC = 0.0

        # dU1/dt
        du1 = (
            1
            / self.parameters["C1"]
            * (current - 1 / self.parameters["R1"] * x[0])
        )
        # dU2/dt
        du2 = (
            1
            / self.parameters["C2"]
            * (current - 1 / self.parameters["R2"] * x[1])
        )

        return np.array([du1, du2, dSoC])

    def get_param(self, name):
        return self.parameters[name]

    # def out(self, sol: OdeSolution):
    #     [u1, u2, soc] = sol.y
    #     ut = self.ocv_curve.soc2ocv(soc) + u1 + u2
    #     return d1
