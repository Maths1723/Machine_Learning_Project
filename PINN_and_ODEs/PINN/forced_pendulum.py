import numpy as np
import torch

# Damped, periodically FORCED pendulum, IC point c varies:
#     y'' + beta y' + sin(y) = A cos(omega t),   y(c) = 0.5, y'(c) = 0,   c in [0, 2]
# The forcing makes the ODE non-autonomous, so y(t; c) is NOT a translate of a
# single trajectory: the time-translation symmetry used by ShiftPINN is broken
# (the solution depends on the forcing phase omega * c at the IC point).

PARAM_NAME_IN_MAT_FILE = 'c_values'

PARAM_MIN_VALUE = 0.0
PARAM_MAX_VALUE = 2.0
NUM_PARAM_VALUES_GEN = 50

FIXED_Y0 = 0.5
FIXED_Y_PRIME0 = 0.0

X_EVAL_START_GEN = 0.0
X_EVAL_END_GEN = 15.0
NUM_X_EVAL_POINTS_GEN = 200

FIXED_BETA = 0.2
FORCING_AMPLITUDE = 0.5
FORCING_OMEGA = 1.2
AUTONOMOUS = False


def _sin(v):
    return torch.sin(v) if torch.is_tensor(v) else np.sin(v)


def _cos(v):
    return torch.cos(v) if torch.is_tensor(v) else np.cos(v)


def ode_system(t, y, param_val_c):
    forcing = FORCING_AMPLITUDE * _cos(FORCING_OMEGA * t)
    return [y[1], -FIXED_BETA * y[1] - _sin(y[0]) + forcing]
