import numpy as np

# Van der Pol with a varying NONLINEARITY parameter (not a varying IC point):
#     y'' = mu * (1 - y^2) * y' - y,     y(0) = 1, y'(0) = 0,   mu in [0.5, 2]
# Unlike the 'c_values' configs, the family is not a translate of a single
# trajectory: both the period and the shape of the limit cycle depend on mu.

PARAM_NAME_IN_MAT_FILE = 'mu_values'

PARAM_MIN_VALUE = 0.5
PARAM_MAX_VALUE = 2.0
NUM_PARAM_VALUES_GEN = 50

FIXED_Y0 = 1.0
FIXED_Y_PRIME0 = 0.0
X_START_IC_POINT = 0.0   # ICs at a fixed time; mu is what varies

X_EVAL_START_GEN = 0.0
X_EVAL_END_GEN = 10.0
NUM_X_EVAL_POINTS_GEN = 200


def ode_system(x, y, mu):
    # pure arithmetic: works for floats, numpy arrays and torch tensors alike
    return [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]
