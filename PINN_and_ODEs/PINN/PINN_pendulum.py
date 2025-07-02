# generalization_pendulum.py

# 1. Add 'import torch' at the top of the file
import torch
import numpy as np 

# These are the parameters for the pendulum ODE
# y'' + beta*y' + alpha*sin(y) = 0
PARAM_NAME_IN_MAT_FILE = 'c_values'
FIXED_ALPHA = 9.81  # Corresponds to g/L
FIXED_BETA = 0.25   # Damping coefficient

# Initial conditions are applied at 'c'
# y(c) = FIXED_Y0
# y'(c) = FIXED_Y_PRIME0
FIXED_Y0 = 0.0
FIXED_Y_PRIME0 = 2.0 

# --- Generation Parameters ---
# Range of the parameter 'c' to generate data for
PARAM_MIN_VALUE = 0.0
PARAM_MAX_VALUE = 2.0
NUM_PARAM_VALUES_GEN = 50

# Range of 'x' (or 't') to evaluate the solution at
X_EVAL_START_GEN = 0.0
X_EVAL_END_GEN = 10.0
NUM_X_EVAL_POINTS_GEN = 100

def ode_system(t, y, param_val_c):
    """
    Defines the second-order ODE for a damped pendulum.
    y is a list or tuple where y[0] is the angle (y) and y[1] is the angular velocity (y').
    """
    dy_dt = y[1]
    
    # This was the problematic line:
    # d2y_dt2 = -FIXED_BETA * y[1] - FIXED_ALPHA * np.sin(y[0])
    
    # 2. This is the corrected line using torch.sin
    d2y_dt2 = -FIXED_BETA * y[1] - FIXED_ALPHA * torch.sin(y[0])
    
    return [dy_dt, d2y_dt2]