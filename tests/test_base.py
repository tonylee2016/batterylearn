import numpy as np
import pandas as pd
import os
from pyens.models import Flywheel, OCV, EcmCell
from pyens.utilities import ivp
from pyens.simulations import Simulator, Data, Current
import matplotlib.pyplot as plt
from scipy import optimize

TESTDATA_FILEPATH = os.path.join(os.path.dirname(__file__), 'CS2_3_9_28_11.csv')


def test_flywheel_base():
    K = np.diag([5, 5, 5, 5, 5])  # add some
    C = np.zeros((5, 5))
    C[0, 1] = 100
    C[1, 0] = -100
    C[0, 0] = 1
    C[1, 1] = 1
    C[2, 2] = 1
    C[3, 3] = 1
    C[4, 4] = 1
    M = np.diag([10, 10, 5, 5, 5])
    fw = Flywheel(K, C, M, w0=50.0, name="fw1")

    v_t = np.arange(0, 10, 0.001)
    x0 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0])

    sol = ivp(fw.ode, x0=x0, v_t=v_t)

    # plt.plot(
    #     sol.t,
    #     np.transpose(
    #         sol.y[
    #         :2,
    #         ]
    #     ),
    # )
    # # plt.legend(["thetax", "thetay", "x", "y", "z"])
    # # plt.show()
    # # plt.plot()


def test_flywheel_model_with_bearing():
    pass


def test_ocv_curve():
    ocv = [3.3, 3.5, 3.55, 3.6, 3.65, 3.68, 3.70, 3.8, 3.95, 4.0, 4.1]
    soc = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    c1 = OCV(name="curve1", ocv=ocv, soc=soc)

    assert c1.ocv == ocv
    assert c1.soc == soc
    # c1.display()


def test_model_run():
    ocv = [3.3, 3.5, 3.55, 3.6, 3.65, 3.68, 3.70, 3.8, 3.95, 4.0, 4.1]
    soc = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    c1 = OCV(name="curve1", ocv=ocv, soc=soc)

    param = {
        "R0": 0.034,
        "R1": 0.022,
        "C1": 1500,
        "R2": 0.019,
        "C2": 65000,
        "CAP": 15,
        "ce": 0.96,
        "v_limits": [2.5, 4.5],
        "SOC_RANGE": [0.0, 100.0],
    }

    dt = 0.1

    initial_soc = 0.0
    CURR_EXCITATION = 7.5
    HOUR = 3600.0
    h_steps = [1.75, 0.25, 0.25, 0.25, 1, 1, 1]
    current_steps = [
        -CURR_EXCITATION,
        -CURR_EXCITATION / 2,
        -CURR_EXCITATION / 4,
        -CURR_EXCITATION / 8,
        0.0,
        CURR_EXCITATION,
        0.0,
    ]

    t_steps = [value * HOUR for value in h_steps]
    total_time = sum(t_steps)
    samples = int(total_time / dt)

    time_np = np.linspace(0.0, total_time, samples)
    step_cur = Current(name="current1")
    for t_step, current_step in zip(t_steps, current_steps):
        step_cur.add_step(current_step, int(t_step / dt))

    data = {"time": time_np, "current": step_cur.current}

    # data = {"time": t_steps, "current": current_steps}

    df = pd.DataFrame(data)
    d1 = Data(name="current_excite", df=df)
    m1 = EcmCell(name="cell_model1", parameters=param, curve=c1)
    # m1.display()
    s1 = Simulator(name="simulator1")
    s1.attach(m1).attach(d1)

    sol = s1.run(
        pair=("cell_model1", "current_excite"),
        x0=np.array([0, 0, initial_soc]),
        config={"solver_type": "adaptive", "solution_name": "sol1"},
    )

    # sol.disp(["current", "soc", "vt"])

    pass


def test_model_fit():
    schema = {
        "Test_Time(s)": "time",
        "Current(A)": "current",
        "Voltage(V)": "vt",
        "rsv_i_dir": True,
    }
    d1 = Data(name='d1', df=None)
    d1.fetch_file(TESTDATA_FILEPATH, schema=schema)
    d1.disp(['current', 'vt'])

    m1 = EcmCell(name="m1")
    s1 = Simulator(name='model_fitter')
    so1 = s1.attach(m1).attach(d1).run(('m1', 'd1'), x0=[0., 0., 100])
    so1.disp(['soc', 'vt', 'current'])
    pass
