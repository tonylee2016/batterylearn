# import matplotlib.pyplot as plt
from scipy.optimize import (
    differential_evolution,
    least_squares,
    minimize,
    shgo,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .simulations import Simulator


class Learner(Simulator):
    """
    input: current, voltage data, SOC-OCV curve, capacity, CE
    output: the R-C parameters fitted to the data"""

    def __init__(self, name):
        Simulator.__init__(self, name=name)

    def fit_parameters(self, names, config, x0, solver, bounds):
        """
        fit the parameters with least_squares
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
        """
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
            CONST_BOUNDS = (
                [i[0] for i in CONST_BOUNDS],
                [i[1] for i in CONST_BOUNDS],
            )
            res = least_squares(
                self.residuals,
                p0,
                bounds=CONST_BOUNDS,
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
        """
        vt: array of terminal voltage from data = data.df.vt.to_numpy()
        p0:init value of parameters
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
        #     res = mean_squared_error(meas_vt, sim_vt, squared=False)
        #     print("rmse", res, len(meas_vt), len(sim_vt))
        #     # ax = d1.df[['vt']].plot()
        #     # d2.df['vt'].plot(ax=ax)
        #     # plt.show()
        #     return res
        # res = abs(meas_vt - sim_vt)
        # print("diff", res)

        # print("rmse", res, len(meas_vt), len(sim_vt))
        return meas_vt - sim_vt
