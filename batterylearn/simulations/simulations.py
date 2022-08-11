# from scipy.integrate._ivp import OdeSolution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from batterylearn.elements import Base, Container
from batterylearn.models import EcmCell
from batterylearn.utilities import ivp


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
    def __init__(self, name, df: pd.DataFrame):
        Base.__init__(self, type="data", name=name)
        self.df = df

    def parse_time(self):
        v_t = self.df["time"]
        t0 = v_t.iloc[0]
        t1 = v_t.iloc[-1]
        return t0, t1, v_t

    def __rvs_cur_dir(self):
        self.df["current"] = -self.df["current"]

    def get_field(self, name):
        return self.df[name]

    def get_current(self, ts):
        return np.interp(ts, self.df["time"], self.df["current"])

    def fetch_file(self, file_path, schema):
        self.df = pd.read_excel(file_path)
        rsv_dir = schema.popitem()
        self.df.rename(columns=schema, inplace=True)
        self.df["time"] = pd.to_datetime(self.df.time)
        if rsv_dir[1]:
            self.__rvs_cur_dir()

    def to_abs_time(self):
        a = self.df["time"] - self.df["time"][0]
        self.df["time"] = a.dt.total_seconds()

    def disp(self, fields=None):
        if fields is None:
            fields = list(self.df.columns)
            fields.remove("time")

        # self.df[['u1','u2','vt','ocv']].plot()
        # self.df[['u1', 'u2']].plot()
        # self.df[['current']].plot()
        # self.df[['soc']].plot()
        figure, axes = plt.subplots(len(fields), 1)
        for idx, field in enumerate(fields):
            self.df.plot(x="time", y=field, ax=axes[idx])
        plt.show()


class Simulator(Base, Container):
    def __init__(self, name):
        Base.__init__(self, type="simulator", name=name)
        Container.__init__(self)
        self.solution = None

    def run(self, pair, x0=None, config=None):
        if x0 is None:
            x0 = [0.0, 0.0, 0]
        if config is None:
            config = {"solver_type": "adaptive", "solution_name": ""}

        model = self.get(pair[0])
        data = self.get(pair[1])

        if not (isinstance(model, EcmCell) and isinstance(data, Data)):
            ValueError("the simulation pair is wrong")

        t0, t1, v_t = data.parse_time()

        # if config["solver_type"] == "adaptive":
        #     v_t = None

        if "max_step" in config:
            max_step = config["max_step"]
        else:
            max_step = np.inf

        sol = ivp(
            fcn=model.ode,
            x0=x0,
            v_t=v_t,
            t_span=(t0, t1),
            force_map=data.get_current,
            method="RK45",
            max_step=max_step,
        )

        # pack the solution
        [u1, u2, soc] = sol.y
        v_t = sol.t
        i_v = data.get_current(v_t)
        ocv = model.ocv_curve.soc2ocv(soc)
        vt = ocv - u1 - u2 - i_v * model.prm("R0")
        df_data = {
            "u1": u1,
            "u2": u2,
            "soc": soc,
            "vt": vt,
            "current": i_v,
            "ocv": ocv,
            "time": v_t,
        }
        df = pd.DataFrame(df_data)
        d2 = Data(name=config["solution_name"], df=df)
        return d2
