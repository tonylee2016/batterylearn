import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame as pd
from scipy.optimize import (
    differential_evolution,
    least_squares,
    leastsq,
    minimize,
)
from sklearn.metrics import mean_squared_error, r2_score

from pyens.models import EcmCell
from pyens.utilities import ivp

from .simulations import Simulator


class Learner(Simulator):
    """
    input: current, voltage data, SOC-OCV curve, capacity, CE
    output: the R-C parameters fitted to the data"""

    def __init__(self, name):
        Simulator.__init__(self, name=name)

    def fit_parameters(self, names, config, x0, method):
        """
        fit the parameters with least_squares
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
        """
        # s_sim = self.get(names[2])
        m_sim = self.get(names[0])
        p0 = [
            m_sim.prm("R0"),
            m_sim.prm("R1"),
            m_sim.prm("C1"),
            m_sim.prm("R2"),
            m_sim.prm("C2"),
            m_sim.prm("CAP"),
        ]
        if method == "ls":
            res = least_squares(
                self.residuals,
                p0,
                # bounds=([0,0,0,0,0],[1,0.5,1600,0.5,65000]),
                method="lm",
                # tr_solver='lsmr',
                args=(names, config, x0, method),
            )
        elif method == "minimize":
            """
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
            """
            res = minimize(
                self.residuals,
                p0,
                # callback=callbackF,
                args=(names, config, x0, method),
                bounds=(
                    (1e-8, 0.3),
                    (1e-8, 10),
                    (90, 50000),
                    (1e-8, 10),
                    (1e-8, 500000),
                    (1e-8, 500),
                ),
                options={
                    "disp": True,
                    "xatol": 1e-8,
                    "fatol": 1e-8,
                    "maxiter": 5000,
                    "maxfev": 5000,
                    # "adaptive": True,
                },
                method="Powell",
                #  Powell, L-BFGS-B, TNC, SLSQP, and trust-constr
            )
        elif method == "global":
            res = differential_evolution(
                func=self.residuals,
                x0=p0,
                # callback=callbackF,
                polish=True,
                args=(names, config, x0, method),
                bounds=(
                    (1e-8, 0.5),
                    (1e-8, 1),
                    (90, 50000),
                    (1e-8, 10),
                    (1e-8, 500000),
                    (1e-8, 500),
                ),
            )
        return res

    # def callbackF(Xi):
    #     global Nfeval
    #     print '{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], Xi[3])
    #     Nfeval += 1

    def residuals(self, p0, names, config, x0, method):
        """
        vt: array of terminal voltage from data = data.df.vt.to_numpy()
        p0:init value of parameters
        """
        # build data and ecm then pass to run function
        m = self.get(names[0])
        prams = {
            "R0": p0[0],
            "R1": p0[1],
            "C1": p0[2],
            "R2": p0[3],
            "C2": p0[4],
            "CAP": p0[5],
            "ce": m.prm("ce"),
            "v_limits": m.prm("v_limits"),
            "SOC_RANGE": m.prm("SOC_RANGE"),
        }
        m.update_rpm(prams)
        d2 = self.run(
            pair=(names[0], names[1]),
            x0=x0,
            config=config,
        )
        d1 = self.get(names[1])
        sim_vt = d2.df.vt
        meas_vt = d1.df.vt
        if method in ["minimize", "global"]:
            res = mean_squared_error(meas_vt, sim_vt, squared=False)
            print("rmse", res, len(meas_vt), len(sim_vt))
            # ax = d1.df[['vt']].plot()
            # d2.df['vt'].plot(ax=ax)
            # plt.show()
            return res
        res = abs(meas_vt - sim_vt)
        print("diff", res)
        return res
