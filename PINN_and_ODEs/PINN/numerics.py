"""Reference (non-NN) solutions used to evaluate the PINNs.

All problems are second-order scalar IVPs written as a first-order system

    y'' = f(t, y, y'; p),     y(t0) = g0,  y'(t0) = g1,

where the initial-condition point t0 may lie anywhere in (or outside) the
evaluation interval. In the 'c_values' configs t0 = c is the varying parameter.
"""

import numpy as np
from scipy.integrate import solve_ivp


def reference_solution(ode_func, g0, g1, t0, t_eval, param=None, rtol=1e-10, atol=1e-12):
    """High-accuracy solution of the IVP evaluated at t_eval.

    Integrates forward and backward from t0 so that t_eval may straddle the
    IC point. Returns an array (2, len(t_eval)) with y and y'.
    """
    t_eval = np.asarray(t_eval, dtype=float)
    param = t0 if param is None else param
    out = np.full((2, t_eval.size), np.nan)

    def rhs(t, y):
        return ode_func(t, y, param)

    for mask, t_end in ((t_eval >= t0, t_eval.max()), (t_eval < t0, t_eval.min())):
        if not mask.any():
            continue
        if t_end == t0:
            out[:, mask] = np.array([[g0], [g1]])
            continue
        sol = solve_ivp(rhs, (t0, t_end), [g0, g1], method="DOP853",
                        rtol=rtol, atol=atol, dense_output=True)
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed from t0={t0} to {t_end}: {sol.message}")
        out[:, mask] = sol.sol(t_eval[mask])
    return out


def relative_l2(y_pred, y_true):
    """Discrete relative L2 error ||y_pred - y_true|| / ||y_true||."""
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=float).ravel()
    return float(np.linalg.norm(y_pred - y_true) / max(np.linalg.norm(y_true), 1e-14))
