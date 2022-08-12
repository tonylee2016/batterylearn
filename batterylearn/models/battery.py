import matplotlib.pyplot as plt
import numpy as np
import schemdraw.elements as elm
from schemdraw import Drawing

from batterylearn.elements import Base, Dynamical


class OCV(Base):

    """_summary_

    Args:
        name (_type_): _description_
        soc (list, optional): _description_. Defaults to [ 3.3, 3.5, 3.55, 3.6, 3.65, 3.68, 3.70, 3.8, 3.95, 4.0, 4.1, ].
        ocv (list, optional): _description_. Defaults to [ 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, ].

    """    """"""
    def __init__(
        self,
        name,
        soc: list = [
            3.3,
            3.5,
            3.55,
            3.6,
            3.65,
            3.68,
            3.70,
            3.8,
            3.95,
            4.0,
            4.1,
        ],
        ocv: list = [
            0.0,
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
            70.0,
            80.0,
            90.0,
            100.0,
        ],
    ):
        Base.__init__(self, type="soc_ocv_curve", name=name)
        self.ocv = ocv
        self.soc = soc

    def display(self):
        """plot the soc-ocv curve
        """
        plt.figure()
        plt.xlabel("soc")
        plt.ylabel("ocv")
        plt.title("curve")
        plt.plot(self.soc, self.ocv)
        plt.show()

    def soc2ocv(self, soc:float) -> float:
        """ocv lookup

        Args:
            soc (float):

        Returns:
            ocv (float):
        """
        return np.interp(soc, self.soc, self.ocv)


class EcmCell(Base, Dynamical):
    """ECM mode for single cell

    Args:
        name (str): name of the cell
        parameters (dict, optional): dict of parameters . Defaults to None.
        curve (OCV, optional): the soc-ocv curve. Defaults to OCV("default").
    """
    def __init__(
        self, name, parameters: dict = None, curve: OCV = OCV("default")
    ):
        Base.__init__(self, type="EMC_Cell_Model", name=name)
        Dynamical.__init__(self)
        if parameters is None:
            parameters = {
                "R0": 0.05,
                "R1": 0.022,
                "tau1": 2,
                "R2": 0.019,
                "tau2": 1,
                "CAP": 1.1,
                "ce": 0.99,
                "v_limits": [1.5, 4.5],
                "SOC_RANGE": [-5.0, 105.0],
            }
        self.__parameters = parameters
        self.ocv_curve = curve

    def update_rpm(self, prm:dict):
        """update thee parameter for the cell

        Args:
            prm (dict): new parameter dictionary 
        """
        self.__parameters = prm

    def prm(self, name:str)->float:
        """fetch parameter

        Args:
            name (str): parameter name

        Returns:
            float: returned value
        """
        return self.__parameters[name]

    def ode(self, t, x, current_series)->np.ndarray:
        """_summary_

        Args:
            t (_type_): t step, used for ode solver only
            x (_type_): state variables, used for ode solver only
            current_series (_type_): external mapping function for current excitation

        Returns:
            np.ndarray: dx
        """

        current = current_series(t)

        # dSoC/dt

        # SOC constrain

        if current > 0 and x[2] > self.prm("SOC_RANGE")[0]:
            dSoC = -1 / self.prm("CAP") * current / 36
        elif current < 0 and x[2] < self.prm("SOC_RANGE")[1]:
            dSoC = -1 / self.prm("CAP") * current * self.prm("ce") / 36
        else:
            dSoC = 0.0

        # dU/dt*C + u/R = i
        du1 = (
            current / self.prm("tau1") / 60.0 * self.prm("R1")
            - x[0] / self.prm("tau1") / 60.0
        )
        # dU2/dt
        du2 = (
            current / self.prm("tau2") / 3600.0 * self.prm("R2")
            - x[0] / self.prm("tau2") / 3600.0
        )

        return np.array([du1, du2, dSoC])

    def out(self, current:np.ndarray, x:np.ndarray)->np.ndarray:
        """output calculation for the cell model

        Args:
            current (np.ndarray): current excitation
            x (np.ndarray): state variables, from the ode solution

        Returns:
            np.ndarray: terminal voltage
        """
        vt = (
            self.ocv_curve.soc2ocv(x[2])
            - x[0]
            - x[1]
            - current * self.prm("R0")
        )
        return vt

    def display(self):
        """display the ECM cell model w/ schemdraw
        """
        with Drawing() as d:
            d.push()
            d += elm.BatteryCell().up()
            d += (
                R0 := elm.Resistor()
                .right()
                .label("{0:.3e}".format(self.prm("R0")))
            )
            d += (
                elm.CurrentLabel(top=False, length=1.0, ofst=0.6)
                .right()
                .at(R0)
            )
            d.push()
            d += elm.Resistor().right().label("{0:.3e}".format(self.prm("R1")))
            d.pop()
            d += elm.Line(l=1.5).down()
            d += (
                elm.Capacitor()
                .right()
                .label("{0:.3e}".format(self.prm("tau1") / self.prm("R1")))
            )
            d += elm.Line(l=1.5).up()
            d += elm.Line(l=1.5).right()
            d.push()
            d += elm.Resistor().right().label("{0:.3e}".format(self.prm("R2")))
            d.pop()
            d += elm.Line(l=1.5).down()
            d += (
                elm.Capacitor()
                .right()
                .label("{0:.3e}".format(self.prm("tau2") / self.prm("R2")))
            )
            d += elm.Line(l=1.5).up()
            d += elm.Line(l=1).right()
            d += elm.Dot().color("blue")
            d.pop()
            d += elm.Line(l=11.5).right()
            d += elm.Dot().color("blue")
