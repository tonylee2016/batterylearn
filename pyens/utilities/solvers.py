from scipy import integrate
import numpy as np


def ivp(fcn, x0, v_t, t_span, force_map=None, method="RK45",max_step=np.inf):
    """
    # numeric solver
    Parameters
    ----------
    :param fcn :  the state-space fcn to be solved .
    :param x0 : the intial state variable condition.
    :param v_t : the time step array.
    :param v_u : the solved state variable.
    :param method:
    :param force_map:
    """
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
     automatic sti‘RK45’ (default), ‘RK23’, ‘DOP853’, ‘Radau’, ‘BDF’, ‘LSODA’:
            """

    y0 = x0
    t_eval = v_t
    if force_map:
        args = (force_map,)
    else:
        args = None
    return integrate.solve_ivp(
        fcn,
        t_span,
        y0,
        method=method,
        t_eval=t_eval,
        dense_output=False,
        events=None,
        vectorized=False,
        args=args,
        max_step=max_step,
    )
