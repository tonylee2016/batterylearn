# from scipy.integrate._ivp import OdeSolution
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from pyens.elements import Base, Container
from pyens.utilities import ivp


class Current(Base):
    def __init__(self, name):
        Base.__init__(self, type="current", name=name)
        self.current = None

    def add_step(self, value, samples):
        if self.current is None:
            self.current = np.linspace(value, value, samples)
        else:
            self.current = np.concatenate(
                (self.current, np.linspace(value, value, samples))
            )
        return self


class Data(Base):
    def __init__(self, name, df: DataFrame):
        Base.__init__(self, type="data", name=name)
        self.df = df

    def parse_time(self):
        v_t = self.df["time"]
        t0 = v_t[0]
        t1 = v_t.iloc[-1]
        return t0, t1, v_t

    def get_field(self, name):
        return self.df[name]

    def get_current(self, ts):
        return np.interp(ts, self.df["time"], self.df["current"])

    def disp(self, fields):
        figure, axes = plt.subplots(len(fields), 1)
        for idx, field in enumerate(fields):
            self.df.plot(x="time", y=field, ax=axes[idx])
        plt.show()


class Simulator(Base, Container):
    def __init__(self, name):
        Base.__init__(self, type="simulator", name=name)
        Container.__init__(self)
        self.solution = None

    def run(self, pair, x0, sol_name):
        model = self.get(pair[0])
        data = self.get(pair[1])

        t0, t1, v_t = data.parse_time()

        sol = ivp(
            fcn=model.ode,
            x0=x0,
            v_t=v_t,
            t_span=(t0, t1),
            force_map=data.get_current,
            method="RK45",
        )

        # pack the solution
        [u1, u2, soc] = sol.y
        i_v = data.get_field("current")
        ocv = model.ocv_curve.soc2ocv(soc)
        ut = ocv - u1 - u2 - data.get_field("current") * model.get_param("R0")
        df_data = {
            "u1": u1,
            "u2": u2,
            "soc": soc,
            "ut": ut,
            "current": i_v,
            "ocv": ocv,
            "time": data.get_field("time"),
        }
        df = DataFrame(df_data)
        d2 = Data(name=sol_name, df=df)
        return d2
