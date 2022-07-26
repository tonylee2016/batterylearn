from pyens.simulations import Learner, Simulator, Data
from pyens.models import EcmCell, OCV
import numpy as np
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import os

TESTDATA_FILEPATH = os.path.join(
    os.path.dirname(__file__), "CALCE_A123_007.csv"
)

schema = {
    "Time": "time",
    "max_current": "current",
    "max_v_tr": "vt",
    "rsv_i_dir": False,
}
d1 = Data(name="d1", df=None)
d1.fetch_file(TESTDATA_FILEPATH, schema=schema)
d1.to_abs_time()

ocv = [
    2.725578,
    2.850545,
    2.952456,
    3.020007,
    3.072155,
    3.114243,
    3.149442,
    3.178211,
    3.196915,
    3.208162,
    3.21203,
    3.252616,
    3.284272,
    3.308415,
    3.311269,
    3.314391,
    3.336869,
    3.345692,
    3.351787,
    3.353211,
    3.354968,
    3.357175,
    3.360247,
    3.364758,
    3.372647,
    3.388452,
    3.421374,
    3.428617,
    3.433137,
    3.438826,
    3.444884,
    3.4514,
    3.458486,
    3.466429,
    3.475249,
    3.5,
]
soc = [
    0.397351,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    98.1,
    98.2,
    98.3,
    98.4,
    98.5,
    98.6,
    98.7,
    98.8,
    100.0,
]

c1 = OCV(name="curve1", ocv=ocv, soc=soc)

c1.display()

param_sim = {
    "R0": 0.034,
    "R1": 0.022,
    "C1": 1500,
    "R2": 0.019,
    "C2": 65000,
    "CAP": 1.1,
    "ce": 0.99,
    "v_limits": [1.5, 4.5],
    "SOC_RANGE": [0.0, 100.0],
}
m_sim = EcmCell(name="cell_model_sim", parameters=param_sim, curve=c1)
l2 = Learner(name="l2")
l2.attach(d1).attach(m_sim)
x0_sim = np.array([0, 0, 100])
config = {"solver_type": "adaptive", "solution_name": "sol1"}
method = "minimize"
res = l2.fit_parameters(("cell_model_sim", "d1"), config, x0_sim, method)

param_exr = {
    "R0": res.x[0],
    "R1": res.x[1],
    "C1": res.x[2],
    "R2": res.x[3],
    "C2": res.x[4],
    "CAP": res.x[5],
    "ce": 0.99,
    "v_limits": [1.5, 4.5],
    "SOC_RANGE": [0.0, 100.0],
}

m_exr = EcmCell(name="cell_model_exr", parameters=param_exr, curve=c1)
s3 = Simulator(name="s3")
s3.attach(d1).attach(m_exr)
d3 = s3.run(("cell_model_exr", "d1"), x0_sim)
ax = d1.df[["vt"]].plot()
d3.df["vt"].plot(ax=ax)
plt.show()
pass
