import matplotlib.pyplot as plt
import numpy as np

from pyens.elements import Base


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


class ECMCell(Base):
    pass
