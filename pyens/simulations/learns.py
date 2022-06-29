from numpy import interp
from pandas import DataFrame as pd
from scipy.optimize import least_squares,minimize
from pyens.elements.elements import Base, Container
from pyens.models import EcmCell
# from pyens.simulations import Data
from pyens.utilities import ivp


class Learner(Base, Container):
    """
    input: current, voltage data, SOC-OCV curve, capacity, CE
    output: the R-C parameters fitted to the data"""

    def __init__(self, name):
        Base.__init__(self, type="learner", name=name)
        Container.__init__(self)

    def fit_parameters(self, p0, data, c1, config, x0,method):
        '''
        fit the parameters with least_squares 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
        '''
        cur_method = data.get_current
        t_array = data.df.time
        if method =="ls":
            res = least_squares(
                self.residuals,
                p0,
                method='lm',
                # tr_solver='lsmr',
                args=(cur_method, t_array, data.df.vt, c1, config, x0),
            )
        elif method =="minimize":
            res = minimize(
                self.residuals,
                p0,
                method='Nelder-Mead',
                args=(cur_method, t_array, data.df.vt, c1, config, x0),
            )
        return res

    def residuals(self, p0, cur_method, t_array, vt, c1, config, x0):
        """
        vt: array of terminal voltage from data = data.df.vt.to_numpy()
        p0:init value of parameters
        """
        sim_vt,sim_time=self.run_sim(p0, cur_method, t_array, c1, config, x0) 
        meas_vt=vt
        if config["solver_type"] == "adaptive": 
            # get vt the same size as estimated vt
            meas_vt=interp(sim_time,t_array,vt)
            
        return meas_vt-sim_vt  

    def run_sim(self, p0, cur_method, t_array, c1, config, x0):
        """
        param v_t : the time step array.
        c1:OCV
        """
        # build ecm based on init parameters
        param_sim = {
            "R0": p0[0],
            "R1": p0[1],
            "C1": p0[2],
            "R2": p0[3],
            "C2": p0[4],
            "CAP": 15,
            "ce": 0.96,
            "v_limits": [2.5, 4.5],
            "SOC_RANGE": [0.0, 100.0],
        }

        m_sim = EcmCell(name="cell_model_sim", parameters=param_sim, curve=c1)
        v_t=t_array
        if config is None:
            config = {"solver_type": "adaptive", "solution_name": ""}

        if config["solver_type"] == "adaptive":
            v_t = None
        sol = ivp(
            fcn=m_sim.ode,
            x0=x0,
            v_t=v_t,
            t_span=(t_array[0], t_array.iloc[-1]),
            force_map=cur_method,
            method="RK45",
        )

        # pack the solution
        [u1, u2, soc] = sol.y
        v_t = sol.t
        i_v = cur_method(v_t)
        ocv = c1.soc2ocv(soc)
        vt = ocv - u1 - u2 - i_v * param_sim["R0"]
        return vt,sol.t
