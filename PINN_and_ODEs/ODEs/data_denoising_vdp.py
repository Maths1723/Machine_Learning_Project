import numpy as np

# Specific fixed coefficient for the Van der Pol equation
FIXED_MU_VALUE = 1.0

PARAM_NAME_IN_MAT_FILE = 'c_values' # 'c' is the varying IC point

PARAM_MIN_VALUE = 0.01  # 'c' can be close to 0, but avoid exactly 0 for some solvers or if the system becomes numerically unstable at 0
PARAM_MAX_VALUE = 2.0
NUM_PARAM_VALUES_GEN = 200

FIXED_Y0 = 0.1       # Fixed y(c) - a small initial perturbation to start oscillations
FIXED_Y_PRIME0 = 0.0 # Fixed y'(c)

# X_START_IC_POINT is NOT defined here because 'c' is the varying IC point.

X_EVAL_START_GEN = 0.0
X_EVAL_END_GEN = 10.0 # A longer interval to observe limit cycle
NUM_X_EVAL_POINTS_GEN = 200 # More points to capture oscillatory behavior

def ode_system(x, y, param_val_c):
    # param_val_c is the 'c' from the ICs, but not used in the RHS of the ODE itself.
    # It's here for API consistency.

    # y[0] is y, y[1] is y'
    dy_dx = y[1]
    d2y_dx2 = FIXED_MU_VALUE * (1 - y[0]**2) * y[1] - y[0]
    return [dy_dx, d2y_dx2]