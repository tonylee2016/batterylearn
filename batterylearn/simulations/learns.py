# import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import (
    differential_evolution,
    least_squares,
    minimize,
    shgo,
)
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .simulations import Simulator


class Learner(Simulator, BaseEstimator):
    """
    a wrapper for model training

    """

    def __init__(self, name):
        """_summary_

        Args:
            name (string): name of the learner
        """
        Simulator.__init__(self, name=name)

    def fit_parameters(
        self,
        names: tuple,
        config: dict,
        x0: np.ndarray,
        solver: str,
        bounds: tuple,
    ):
        """_summary_

        Args:
            names (tuple): init value of parameters
            config (dict): name of the model and datasets for simulation
            x0 (np.array): init value of simulation
            solver (str): solver type ['diff','minimize']
            bounds (tuple): bounds for the optimizer

        Returns:
            solution: fitting solutions
        """ """"""

        # s_sim = self.get(names[2])
        m_sim = self.get(names[0])

        CONST_BOUNDS = (
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
        )

        # first to pack p0
        keys = ["R0", "R1", "tau1", "R2", "tau2", "CAP"]

        p0 = list(
            map(
                lambda x, y: (m_sim.prm(x) - y[0]) / (y[1] - y[0]),
                keys,
                bounds,
            )
        )

        if solver == "ls":
            CONST_BOUNDS_1 = (
                [i[0] for i in CONST_BOUNDS],
                [i[1] for i in CONST_BOUNDS],
            )
            res = least_squares(
                self.residuals,
                p0,
                bounds=CONST_BOUNDS_1,
                args=(names, config, x0, solver, bounds),
                verbose=2,
                xtol=None,
            )
        elif solver == "minimize":
            """
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
            """
            res = minimize(
                self.residuals,
                p0,
                args=(names, config, x0, solver, bounds),
                method=config["method"],
                bounds=CONST_BOUNDS,
                options={
                    "disp": True,
                    "maxiter": config["maxiter"],
                    "maxfev": config["maxfev"],
                    # "adaptive": True,
                },
                #  Powell, L-BFGS-B, TNC, SLSQP, and trust-constr
            )

        elif solver == "diff":
            res = differential_evolution(
                func=self.residuals,
                x0=p0,
                polish=True,
                args=(names, config, x0, solver, bounds),
                disp=True,
                bounds=CONST_BOUNDS,
                workers=-1,
                maxiter=config["maxiter"],
            )
        elif solver == "shgo":
            res = shgo(
                func=self.residuals,
                bounds=CONST_BOUNDS,
                args=(names, config, x0, solver, bounds),
            )
        return res

    def residuals(self, p0, names, config, x0, method, bounds):
        """_summary_

        Args:
            p0 (list): init value of parameters
            names (tuple): name of the model and datasets for simulation
            config (dict): simulation configuration
            x0 (list): init value of simulation
            method (string): optimization method
            bounds (tuple): optmizer bounds

        Returns:
            np.array: fitting error
        """

        # build data and ecm then pass to run function
        m = self.get(names[0])

        # p0 is normalized, so we unpack it
        # _R0 = (R0 - bounds[0][0])/(bounds[0][1] - bounds[0][0])
        # R0 = (bounds[0][1] - bounds[0][0])*_R0 + bounds[0][0]

        a = list(map(lambda x, y: (y[1] - y[0]) * x + y[0], p0, bounds))

        prams = {
            "R0": a[0],
            "R1": a[1],
            "tau1": a[2],
            "R2": a[3],
            "tau2": a[4],
            "CAP": a[5],
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
        if method in ["minimize", "diff"]:
            res = mean_squared_error(meas_vt, sim_vt)
            if method in ["minimize"]:
                print("rmse", res, len(meas_vt), len(sim_vt))
            return res
        return meas_vt - sim_vt
