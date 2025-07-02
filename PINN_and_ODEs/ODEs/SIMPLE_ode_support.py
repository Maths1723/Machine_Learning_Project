# SIMPLE_ode_support.py (CORRECTED FINAL VERSION)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import scipy.io # To load .mat files
from scipy.integrate import solve_ivp # For numerical solving in Python

# --- Helper Functions ---
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available(): # For Apple Silicon Macs
        return torch.device("mps")
    return torch.device("cpu")

def get_user_choice(prompt, valid_choices):
    """
    Prompts the user for a choice and validates it against a list of valid options.

    Args:
        prompt (str): The message to display to the user.
        valid_choices (list): A list of valid single-character choices (e.g., ['g', 'n', 'd']).

    Returns:
        str: The validated user choice.
    """
    while True:
        choice = input(prompt).lower()
        if choice in valid_choices:
            return choice
        print(f"Invalid choice. Please enter one of: {', '.join(valid_choices)}.")


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# --- Custom Dataset for ODE Solutions ---
class ODESolutionDataset(Dataset):
    def __init__(self, mat_file_path, input_param_name):
        data = scipy.io.loadmat(mat_file_path)
        self.input_param_values = torch.from_numpy(data[input_param_name]).float()  
        self.solution_vectors = torch.from_numpy(data['solution_vectors']).float()
        self.x_evaluation_points = data['x_evaluation_points'].flatten()  
        
        # Initial conditions fixed for the entire dataset
        # Ensure these match how data was generated (GLOBAL_Y0_FIXED_AT_C, GLOBAL_Y_PRIME0_FIXED_AT_C)
        self.y0 = data['y0'].item()  
        self.y_prime0 = data['y_prime0'].item()  

    def __len__(self):
        return len(self.input_param_values)

    def __getitem__(self, idx):
        input_param = self.input_param_values[idx]  
        target_solution = self.solution_vectors[idx]
        return input_param, target_solution

# --- Neural Network for ODE Solution ---
class ODE_MLP(nn.Module):
    def __init__(self, input_dim=1, output_dim=100, hidden_layers=3, hidden_dim=128):
        super(ODE_MLP, self).__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# --- Custom Loss Function for ODE ---
class CustomODELoss(nn.Module):
    def __init__(self, alpha, beta, x_eval_points, y0_fixed_at_c, y_prime0_fixed_at_c): # <-- THESE ARE THE CRITICAL ARGUMENTS
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.x_eval_points_tensor = torch.from_numpy(x_eval_points).float()
        self.y0_fixed_at_c = y0_fixed_at_c  
        self.y_prime0_fixed_at_c = y_prime0_fixed_at_c  

        # Pre-calculate dx for numerical differentiation
        if len(self.x_eval_points_tensor) > 1:
            self.dx = self.x_eval_points_tensor[1] - self.x_eval_points_tensor[0]
        else:
            # Handle case with single x_eval_point (though less typical for ODEs)
            self.dx = torch.tensor(1.0, device=self.x_eval_points_tensor.device)


    def forward(self, predicted_solution_batch, target_solution_batch, c_input_batch):
        # Move x_eval_points_tensor to the same device as the predictions for calculations
        x_eval_points_on_device = self.x_eval_points_tensor.to(predicted_solution_batch.device)

        # Term 1: Integral term (approximated by MSE over the solution vector)
        integral_term = torch.mean((predicted_solution_batch - target_solution_batch)**2)

        # Terms 2 & 3: Boundary conditions F_p(c) - y0 and F_p'(c) - y'0
        # We need to find the index in x_eval_points closest to each c_val in the batch.
        
        fp_c_batch = []
        fp_prime_c_batch = []

        for i in range(c_input_batch.shape[0]):
            c_val = c_input_batch[i].item() # Extract the scalar c value for this sample
            predicted_sol_for_c = predicted_solution_batch[i] # Predicted solution vector for this c

            # Find the index in x_eval_points_on_device closest to c_val
            closest_idx = torch.argmin(torch.abs(x_eval_points_on_device - c_val)).item()

            # F_p(c): Value of the predicted solution at or near 'c'
            fp_c = predicted_sol_for_c[closest_idx]
            fp_c_batch.append(fp_c)

            # F_p'(c): Numerical derivative at or near 'c'
            # Simple central difference for approximation (or forward/backward at ends)
            if closest_idx == 0: # Forward difference at the start
                if predicted_sol_for_c.shape[0] > 1:
                    fp_prime_c = (predicted_sol_for_c[1] - predicted_sol_for_c[0]) / self.dx
                else: # Only one point, derivative is ill-defined (or dx is 1.0 if only 1 point)
                    fp_prime_c = torch.tensor(0.0, device=predicted_sol_for_c.device) # Default to 0 derivative
            elif closest_idx == predicted_sol_for_c.shape[0] - 1: # Backward difference at the end
                fp_prime_c = (predicted_sol_for_c[-1] - predicted_sol_for_c[-2]) / self.dx
            else: # Central difference
                fp_prime_c = (predicted_sol_for_c[closest_idx + 1] - predicted_sol_for_c[closest_idx - 1]) / (2 * self.dx)
            fp_prime_c_batch.append(fp_prime_c)

        # Convert lists to tensors
        fp_c_batch_tensor = torch.stack(fp_c_batch)
        fp_prime_c_batch_tensor = torch.stack(fp_prime_c_batch)

        # Term 2: Alpha * [F_p(c) - y_0]^2
        term2 = self.alpha * torch.mean((fp_c_batch_tensor - self.y0_fixed_at_c)**2)

        # Term 3: Beta * [F_p'(c) - y'_0]^2
        term3 = self.beta * torch.mean((fp_prime_c_batch_tensor - self.y_prime0_fixed_at_c)**2)

        total_loss = integral_term + term2 + term3
        return total_loss

# --- Training and Testing Functions ---
def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (param_input, target_solution) in enumerate(dataloader):
        param_input, target_solution = param_input.to(device), target_solution.to(device)

        # Compute prediction error
        pred = model(param_input)
        loss = loss_fn(pred, target_solution, param_input) # Pass param_input (which is 'c') to loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(param_input)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for param_input, target_solution in dataloader:
            param_input, target_solution = param_input.to(device), target_solution.to(device)
            pred = model(param_input)
            test_loss += loss_fn(pred, target_solution, param_input).item() # Pass param_input (which is 'c') to loss
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

# --- Numerical ODE Solver (for plotting comparison) ---
def solve_ode_numerically(ode_func, initial_conditions_y0_yprime0, x_start_for_ic, x_end_for_integration, x_eval_points_to_interpolate):
    """
    Solves the ODE numerically using scipy.integrate.solve_ivp.
    This version correctly handles initial conditions at a specific x_start_for_ic.
    
    :param ode_func: Function handle for the ODE (t, y) -> dy/dt
    :param initial_conditions_y0_yprime0: [y(x_start_for_ic), y'(x_start_for_ic)]
    :param x_start_for_ic: The specific x-value where initial_conditions are applied (this is 'c').
    :param x_end_for_integration: The final x-value for integration (e.g., x_eval_points[-1]).
    :param x_eval_points_to_interpolate: Points at which to evaluate the solution for output.
    :return: Interpolated solution for y(x) at x_eval_points_to_interpolate.
    """
    
    # Integrate forward from c to x_eval_points_to_interpolate.max()
    sol_fwd = solve_ivp(ode_func, [x_start_for_ic, x_eval_points_to_interpolate[-1]], initial_conditions_y0_yprime0,  
                        method='RK45', rtol=1e-6, atol=1e-8, dense_output=True)
    
    # Integrate backward from c to x_eval_points_to_interpolate.min() (if needed)
    sol_bwd = None
    if x_start_for_ic > x_eval_points_to_interpolate[0]:
        sol_bwd = solve_ivp(ode_func, [x_start_for_ic, x_eval_points_to_interpolate[0]], initial_conditions_y0_yprime0,  
                            method='RK45', rtol=1e-6, atol=1e-8, dense_output=True)
        
    # Combine and interpolate
    if sol_bwd and sol_bwd.success: # Check if backward integration was successful
        # Get y-values from backward solution (excluding the starting point to avoid duplicates)
        x_bwd_eval = x_eval_points_to_interpolate[x_eval_points_to_interpolate < x_start_for_ic]
        if x_bwd_eval.size > 0:
            y_bwd = sol_bwd.sol(x_bwd_eval)[0, :]  
        else:
            y_bwd = np.array([])
        
        # Get y-values from forward solution (including the starting point if it's in x_eval_points)
        x_fwd_eval = x_eval_points_to_interpolate[x_eval_points_to_interpolate >= x_start_for_ic]
        if x_fwd_eval.size > 0:
            y_fwd = sol_fwd.sol(x_fwd_eval)[0, :]  
        else:
            y_fwd = np.array([])

        # Concatenate x and y values
        x_combined = np.concatenate((x_bwd_eval, x_fwd_eval))
        y_combined = np.concatenate((y_bwd, y_fwd))
        
        # Sort by x values (important for interpolation)
        sort_indices = np.argsort(x_combined)
        x_combined_sorted = x_combined[sort_indices]
        y_combined_sorted = y_combined[sort_indices]
        
        # Interpolate onto the desired x_eval_points
        interpolated_solution = np.interp(x_eval_points_to_interpolate, x_combined_sorted, y_combined_sorted)
        return interpolated_solution
    else:
        # Only forward integration needed OR backward integration failed
        # Directly evaluate the dense output solution at the desired points
        if sol_fwd.success: # Ensure forward integration was successful
            interpolated_solution = sol_fwd.sol(x_eval_points_to_interpolate)[0, :]  
            return interpolated_solution
        else:
            # If both fail, return NaNs or zeros for the solution to indicate failure
            print(f"Warning: solve_ivp failed for x_start_for_ic={x_start_for_ic}. Returning NaNs.")
            return np.full_like(x_eval_points_to_interpolate, np.nan)