from pandas import DataFrame as pd
from pyens.elements.elements import Container,Base
from scipy.optimize import least_squares
from pyens.simulations import Data as Data
from pyens.models import EcmCell
from pyens.utilities import ivp

class Learner(Base, Container):
    '''
    input: current, voltage data, SOC-OCV curve, capacity, CE
    output: the R-C parameters fitted to the data
'''
    def __init__(self,name):
        Base.__init__(self,type="learner",name=name)
        Container.__init__(self)
    
    def fit_parameters(self,p0,data,c1,config,x0):
        cur_method=data.get_current
        t_array=data.df.time
        res=least_squares(self.residuals,p0, args=(cur_method,t_array,data.df.vt,c1,config,x0))
        return res
        
    def residuals(self,p0,cur_method,t_array,vt,c1,config,x0):
        '''
        vt: array of terminal voltage from data = data.df.vt.to_numpy()
        p0:init value of parameters 
        '''
        
        res=vt-self.run_sim(p0,cur_method,t_array,c1,config,x0)
        return res

    def run_sim(self,p0,cur_method,t_array,c1,config,x0):
        '''
        param v_t : the time step array.
        c1:OCV 
        '''
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

        if config is None:
            config = {"solver_type": "adaptive", "solution_name": ""}

        if config["solver_type"] == "adaptive":
            v_t = None
        sol = ivp(
            fcn=m_sim.ode,
            x0=x0,
            v_t=t_array,
            t_span=(t_array[0],t_array.iloc[-1]),
            force_map=cur_method,
            method="RK45",
        )

        # pack the solution
        [u1, u2, soc] = sol.y
        v_t = sol.t
        i_v = cur_method(v_t)
        ocv = c1.soc2ocv(soc)
        vt = ocv - u1 - u2 - i_v * param_sim["R0"]
        return vt
        
