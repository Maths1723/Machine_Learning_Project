import numpy as np

# Non-linear pendulum with damping (attrition), no forcing:
# y'' + beta * y' + alpha * sin(y) = 0
# The varying parameter is 'c' from the initial conditions y(c)=y0 and y'(c)=y'0.

PARAM_NAME_IN_MAT_FILE = 'c_values' # 'c' is the varying initial condition point

PARAM_MIN_VALUE = 0.0 # 'c' can start from 0 for this ODE
PARAM_MAX_VALUE = 2.0
NUM_PARAM_VALUES_GEN = 8

FIXED_Y0 = 0.5 # Fixed initial angle
FIXED_Y_PRIME0 = 0.0 # Fixed initial angular velocity
# X_START_IC_POINT is NOT defined here because 'c' is the varying IC point.

X_EVAL_START_GEN = 0.0 # Evaluation starts from 0, regardless of 'c'
X_EVAL_END_GEN = 15.0 # Evaluate over a longer time to observe damping behavior
NUM_X_EVAL_POINTS_GEN = 200 # More points for smoother curves

# Fixed coefficients for the ODE
FIXED_BETA = 0.2 # Damping (attrition) coefficient (choose a value to show noticeable damping)
FIXED_ALPHA = 1.0 # Restoring force coefficient (e.g., g/L for pendulum)

def ode_system(t, y, param_val_c):
    # param_val_c is the 'c' from the ICs, but not used in the RHS of the ODE itself.
    # It's here for API consistency.
    
    # y[0] is y (angle), y[1] is y' (angular velocity)
    
    dy_dt = y[1]
    # d2y_dt2 = -beta * y' - alpha * sin(y)
    d2y_dt2 = -FIXED_BETA * y[1] - FIXED_ALPHA * np.sin(y[0])
    
    return [dy_dt, d2y_dt2]