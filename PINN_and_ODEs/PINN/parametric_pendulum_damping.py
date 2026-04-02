import numpy as np
import torch

# Damped pendulum with varying damping coefficient:
#     y'' + beta * y' + sin(y) = 0,     y(0) = 1, y'(0) = 0,   beta in [0.05, 0.6]
# From almost conservative (beta ~ 0) to strongly damped.

PARAM_NAME_IN_MAT_FILE = 'beta_values'

PARAM_MIN_VALUE = 0.05
PARAM_MAX_VALUE = 0.6
NUM_PARAM_VALUES_GEN = 50

FIXED_Y0 = 1.0
FIXED_Y_PRIME0 = 0.0
X_START_IC_POINT = 0.0

X_EVAL_START_GEN = 0.0
X_EVAL_END_GEN = 15.0
NUM_X_EVAL_POINTS_GEN = 200

FIXED_ALPHA = 1.0  # g / L


def _sin(v):
    return torch.sin(v) if torch.is_tensor(v) else np.sin(v)


def ode_system(x, y, beta):
    return [y[1], -beta * y[1] - FIXED_ALPHA * _sin(y[0])]
