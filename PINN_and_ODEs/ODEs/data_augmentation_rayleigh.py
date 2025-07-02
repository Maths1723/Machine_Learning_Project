import numpy as np

# Problem-Specific Parameters
PARAM_NAME_IN_MAT_FILE = 'c_values' # 'c' is the varying initial condition point

PARAM_MIN_VALUE = 0.0
PARAM_MAX_VALUE = 2.0
NUM_PARAM_VALUES_GEN = 8

FIXED_Y0 = 1.0
FIXED_Y_PRIME0 = 0.0
# X_START_IC_POINT is NOT defined because 'c' is the varying IC point.

X_EVAL_START_GEN = 0.0
X_EVAL_END_GEN = 10.0
NUM_X_EVAL_POINTS_GEN = 100

# Fixed coefficients for the Rayleigh equation
FIXED_MU = 1.0

def ode_system(t, y, param_val_c):
    # The Rayleigh equation is given as x'' + mu*( (1/3)*x'^3 - x' ) + x = 0
    # Rearranging to solve for x'':
    # x'' = -mu*( (1/3)*x'^3 - x' ) - x
    # We convert it to a system of two first-order ODEs:
    # Let y[0] = x (position)
    # Let y[1] = x' (velocity)
    # Then:
    # y[0]' = y[1]
    # y[1]' = -FIXED_MU * ((1/3) * y[1]**3 - y[1]) - y[0]

    dy_dt = y[1]
    d2y_dt2 = -FIXED_MU * ((1/3) * (y[1]**3) - y[1]) - y[0]
    
    # param_val_c is the 'c' from the ICs, but not used in the RHS of the ODE itself.
    # It's here for API consistency.
    
    return [dy_dt, d2y_dt2]