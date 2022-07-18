from pyens.simulations import Learner,Simulator,Data
from pyens.models import EcmCell, OCV
import numpy as np
from pandas import DataFrame as df
import os


TESTDATA_FILEPATH = os.path.join(
    os.path.dirname(__file__), "CS2_3_9_28_11.csv"
)

schema = {
    "Test_Time(s)": "time",
    "Current(A)": "current",
    "Voltage(V)": "vt",
    "rsv_i_dir": True,
}
d1 = Data(name='d1', df=None)
d1.fetch_file(TESTDATA_FILEPATH, schema=schema)

ocv = [3.3, 3.5, 3.55, 3.6, 3.65, 3.68, 3.70, 3.8, 3.95, 4.0, 4.1]
soc = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

c1 = OCV(name="curve1", ocv=ocv, soc=soc)

c1.display()

param_sim = {
 "R0": 0.034,
 "R1": 0.022,
 "C1": 1500,
 "R2": 0.019,
 "C2": 65000,
 "CAP": 1.15,
 "ce": 0.96,
 "v_limits": [2.5, 4.5],
 "SOC_RANGE": [0.0, 100.0],
}
m_sim = EcmCell(name="cell_model_sim", parameters=param_sim, curve=c1)
l2=Learner(name="l2")
l2.attach(d1).attach(m_sim)
x0_sim=np.array([0, 0, 100])
config={"solver_type": None, "solution_name": "sol1"}
method="global"
res=l2.fit_parameters(("cell_model_sim","d1"),config,x0_sim,method)