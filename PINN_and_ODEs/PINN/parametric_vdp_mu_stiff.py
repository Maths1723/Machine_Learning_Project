import numpy as np

# Stiffer Van der Pol family: relaxation oscillations develop as mu grows.
#     y'' = mu (1 - y^2) y' - y,   y(0) = 1, y'(0) = 0,   mu in [2, 5],  t in [0, 20]
# Period ~ (3 - 2 ln 2) mu for large mu; sharp fronts of width ~ 1/mu.

PARAM_NAME_IN_MAT_FILE = 'mu_values'

PARAM_MIN_VALUE = 2.0
PARAM_MAX_VALUE = 5.0
NUM_PARAM_VALUES_GEN = 50

FIXED_Y0 = 1.0
FIXED_Y_PRIME0 = 0.0
X_START_IC_POINT = 0.0

X_EVAL_START_GEN = 0.0
X_EVAL_END_GEN = 20.0
NUM_X_EVAL_POINTS_GEN = 400


def ode_system(x, y, mu):
    return [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]
