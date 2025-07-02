# main_ode_trainer.py (FINAL VERSION with corrected plot_denoiser_flux to only show NN output with varied colors and added error norm plotting)
# --- MODIFIED TO INCLUDE NO-DATA (PINN) TRAINING ---
# - - - libs - - -
import os
import importlib.util
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy.io  # To save .mat files
import re # For parsing noise std from filename

# It's assumed that a 'SIMPLE_ode_support.py' file exists with these helper functions.
# For this script to be self-contained, placeholder definitions are provided here.
# --- Placeholder Support Functions ---
def get_device():
    """Gets the best available device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_user_choice(prompt, valid_choices):
    """Prompts the user for input and validates it against a list of choices."""
    while True:
        choice = input(prompt).lower()
        if choice in valid_choices:
            return choice
        print(f"Invalid choice. Please select from {valid_choices}.")

def save_model(model, path):
    """Saves the model's state dictionary."""
    print(f"Saving model to '{path}'...")
    torch.save(model.state_dict(), path)
    print("Model saved.")

def solve_ode_numerically(ode_func, initial_conditions, x_start_ic, x_end, x_eval_points):
    """
    Solves a 2nd order ODE using SciPy's solve_ivp, handling integration direction.
    """
    from scipy.integrate import solve_ivp
    
    # solve_ivp requires t_eval to be sorted.
    t_span = [x_start_ic, x_end]
    
    # If integrating backward, reverse the span for the solver
    if x_start_ic > x_end:
        t_span_for_solver = [x_end, x_start_ic]
    else:
        t_span_for_solver = [x_start_ic, x_end]

    sol = solve_ivp(
        fun=ode_func,
        t_span=t_span_for_solver,
        y0=initial_conditions,
        t_eval=sorted(x_eval_points),
        dense_output=True,
        method='RK45'
    )
    
    # The output of the solver is always sorted by time.
    # We need to re-order it to match the original x_eval_points if they were in descending order.
    # This is a simplified approach; a more robust solution might involve interpolation.
    y_solution = sol.sol(x_eval_points)
    return y_solution[0]

# --- Placeholder Classes from SIMPLE_ode_support ---

class ODESolutionDataset(torch.utils.data.Dataset):
    """Dataset for loading ODE solutions from .mat files."""
    def __init__(self, mat_file_path, param_name):
        data = scipy.io.loadmat(mat_file_path)
        self.input_param_values = torch.tensor(data[param_name], dtype=torch.float32)
        self.solution_vectors = torch.tensor(data['solution_vectors'], dtype=torch.float32)
        self.x_evaluation_points = data['x_evaluation_points'].flatten()
        self.y0 = data['y0'].item()
        self.y_prime0 = data['y_prime0'].item()

    def __len__(self):
        return len(self.input_param_values)

    def __getitem__(self, idx):
        return self.input_param_values[idx], self.solution_vectors[idx]

class ODE_MLP(nn.Module):
    """Standard MLP for mapping parameter 'p' to a full solution vector."""
    def __init__(self, input_dim=1, output_dim=100, hidden_layers=4, hidden_units=128):
        super(ODE_MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_units), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_units, hidden_units), nn.ReLU()])
        layers.append(nn.Linear(hidden_units, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, p):
        return self.net(p)

class CustomODELoss(nn.Module):
    """Placeholder for a custom loss function (if any)."""
    def __init__(self, alpha, beta, x_eval_points, y0_fixed_at_c, y_prime0_fixed_at_c):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, predicted, target):
        return self.mse(predicted, target)

def train(loader, model, criterion, optimizer, device):
    """Standard training loop for data-driven models."""
    model.train()
    total_loss = 0
    for params, solutions in loader:
        params, solutions = params.to(device), solutions.to(device)
        optimizer.zero_grad()
        predicted = model(params)
        loss = criterion(predicted, solutions)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"    Train Loss: {avg_loss:.6f}")

def test(loader, model, criterion, device):
    """Standard testing loop for data-driven models."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for params, solutions in loader:
            params, solutions = params.to(device), solutions.to(device)
            predicted = model(params)
            loss = criterion(predicted, solutions)
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"    Test Loss: {avg_loss:.6f}")


# Set environment variables
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# - - - Global Variables Initialization - - -
mat_data_path = ""
ode_func_file_path = ""
input_param_name_in_mat = ""
ode_function = None

GLOBAL_Y0_FIXED_AT_C = None
GLOBAL_Y_PRIME0_FIXED_AT_C = None
FIXED_IC_POINT = None
PARAM_MIN_LOADED = None
PARAM_MAX_LOADED = None
NUM_PARAM_VALUES_LOADED = None
X_EVAL_START_LOADED = None
X_EVAL_END_LOADED = None
NUM_X_EVAL_POINTS_LOADED = None

device = get_device() # Initialize device globally


# --- NEW: PINN/No-Data Training Components ---

class PINN_MLP(nn.Module):
    """
    Neural Network for Physics-Informed training.
    It takes time (t) and a parameter (p) as input and outputs the solution y(t).
    """
    def __init__(self, input_dim=2, output_dim=1, hidden_layers=4, hidden_units=64):
        super(PINN_MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_units, hidden_units), nn.Tanh()])
        layers.append(nn.Linear(hidden_units, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t, p):
        # Concatenate t and p to form the input for the network
        input_tensor = torch.cat([t, p], dim=1)
        return self.net(input_tensor)

class NoDataLoss(nn.Module):
    r"""
    Custom loss function for training without pre-generated data (PINN).
    Calculates the loss based on the ODE's residual and boundary conditions.
    Loss C(p) = \int_0^1 [ \Phi[F_p(t)] ]^2 dt + \alpha [ F_p(c) - g_0 ]^2 + \beta [ F_p'(c) - g_1 ]^2
    """
    def __init__(self, ode_func, alpha, beta, g0, g1, device):
        super().__init__()
        self.ode_func = ode_func  # The function defining the ODE dynamics
        self.alpha = alpha
        self.beta = beta
        self.g0 = g0  # Boundary condition value y(c)
        self.g1 = g1  # Boundary condition value y'(c)
        self.device = device

    def forward(self, model, p):
        # p is the parameter 'c' which is also the point for the boundary conditions
        p.requires_grad_(True)

        # 1. Physics Residual Loss (Integral Term)
        collocation_points_t = torch.linspace(0, 1, 101, device=self.device).reshape(-1, 1)
        collocation_points_t.requires_grad_(True)
        
        param_tensor_for_model = p.expand_as(collocation_points_t)

        F_p = model(collocation_points_t, param_tensor_for_model)

        F_p_t = torch.autograd.grad(F_p, collocation_points_t, grad_outputs=torch.ones_like(F_p), create_graph=True)[0]
        F_p_tt = torch.autograd.grad(F_p_t, collocation_points_t, grad_outputs=torch.ones_like(F_p_t), create_graph=True)[0]
        
        y_vec_for_ode = [F_p, F_p_t]
        y_double_prime_from_ode = self.ode_func(collocation_points_t, y_vec_for_ode, p)[1]
        
        physics_residual = F_p_tt - y_double_prime_from_ode
        loss_physics = torch.mean(physics_residual**2)

        # 2. Boundary Condition Loss at t=c
        t_boundary = p.clone().reshape(1,1)
        t_boundary.requires_grad_(True)
        param_tensor_for_bc = p.clone().reshape(1,1)
        
        F_p_at_c = model(t_boundary, param_tensor_for_bc)
        F_p_prime_at_c = torch.autograd.grad(F_p_at_c, t_boundary, grad_outputs=torch.ones_like(F_p_at_c), create_graph=True)[0]

        loss_bc1 = (F_p_at_c - self.g0)**2
        loss_bc2 = (F_p_prime_at_c - self.g1)**2

        # 3. Total Loss
        total_loss = loss_physics + self.alpha * loss_bc1 + self.beta * loss_bc2
        return total_loss.squeeze()


def train_with_no_data():
    """
    Main function to handle the training of a PINN model without any data.
    """
    print("\n--- Physics-Informed Training (No Data) ---")

    if input_param_name_in_mat != 'c_values':
        print(f"Warning: No-data training is designed for problems where the parameter defines the IC point (e.g., 'c_values').")
        print(f"Current parameter is '{input_param_name_in_mat}'. The results might not be meaningful.")

    try:
        alpha = float(input("Enter weight for BC y(c) (alpha, e.g., 100.0): "))
        beta = float(input("Enter weight for BC y'(c) (beta, e.g., 100.0): "))
        epochs = int(input("Enter number of training epochs (e.g., 5000): "))
        lr = float(input("Enter learning rate (e.g., 1e-3): "))
        num_params_to_train = int(input(f"How many parameter values to train on in range [{PARAM_MIN_LOADED}, {PARAM_MAX_LOADED}]? (e.g., 50) (for error norm graphical output): "))
        if alpha < 0 or beta < 0 or epochs <= 0 or lr <= 0 or num_params_to_train <= 0:
            raise ValueError("All inputs must be positive.")
    except (ValueError, TypeError) as e:
        print(f"Invalid input. Please enter valid numbers. Error: {e}")
        return 'back_to_main'

    pinn_model = PINN_MLP().to(device)
    optimizer = torch.optim.Adam(pinn_model.parameters(), lr=lr)
    
    loss_fn = NoDataLoss(ode_func=ode_function, alpha=1, beta=1, 
                         g0=GLOBAL_Y0_FIXED_AT_C, g1=GLOBAL_Y_PRIME0_FIXED_AT_C, device=device)

    print("Starting PINN training...")
    param_range = torch.linspace(PARAM_MIN_LOADED, PARAM_MAX_LOADED, num_params_to_train, device=device)

    for epoch in range(epochs):
        pinn_model.train()
        total_epoch_loss = 0
        
        for p_val in param_range:
            p_tensor = p_val.reshape(1, 1)
            
            optimizer.zero_grad()
            loss = loss_fn(pinn_model, p_tensor)
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            avg_loss = total_epoch_loss / len(param_range)
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.6f}")

    print("PINN training complete.")
    model_path = f"pinn_model_{input_param_name_in_mat.replace('_values','')}.pth"
    save_model(pinn_model, model_path)

    return pinn_post_train_menu(pinn_model, loss_fn, param_range)


def pinn_post_train_menu(model, loss_fn, trained_param_range):
    """Presents a menu for actions after PINN training."""
    while True:
        print("\n--- PINN Model Actions ---")
        choice = get_user_choice(
            "What do you want to do? (s)plot single solution, (f)lot flux, (e)rror norm, (b)ack to main menu, (q)uit? [s/f/e/b/q]: ",
            ['s', 'f', 'e', 'b', 'q']
        )

        if choice == 's':
            try:
                c_val_str = input(f"Enter a value for parameter 'c' (trained range: [{PARAM_MIN_LOADED:.2f}, {PARAM_MAX_LOADED:.2f}]): ")
                c_val = float(c_val_str)
                plot_pinn_single_solution(model, c_val)
            except ValueError:
                print("Invalid input. Please enter a number.")

        elif choice == 'f':
            try:
                num_lines = int(input("Enter number of flux lines to plot (e.g., 10): "))
                if num_lines > 0:
                    plot_pinn_flux(model, num_lines)
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

        elif choice == 'e':
            plot_pinn_error_norm(model, loss_fn, trained_param_range)

        elif choice == 'b':
            return 'back_to_main'
        elif choice == 'q':
            return 'quit_program'


def plot_pinn_single_solution(model, c_val):
    """Plots a single solution from the PINN model for a given parameter 'c'."""
    print(f"\n--- Plotting PINN Solution for c = {c_val:.3f} ---")
    model.eval()
    
    t_eval = np.linspace(X_EVAL_START_LOADED, X_EVAL_END_LOADED, NUM_X_EVAL_POINTS_LOADED)
    t_tensor = torch.tensor(t_eval, dtype=torch.float32).reshape(-1, 1).to(device)
    p_tensor = torch.tensor([[c_val]], dtype=torch.float32).expand_as(t_tensor).to(device)
    
    with torch.no_grad():
        predicted_solution = model(t_tensor, p_tensor).cpu().numpy()

    ode_func_scipy = lambda t, y: ode_function(t, y, c_val)
    ic = [GLOBAL_Y0_FIXED_AT_C, GLOBAL_Y_PRIME0_FIXED_AT_C]
    numerical_solution = solve_ode_numerically(ode_func_scipy, ic, c_val, t_eval[-1], t_eval)

    plt.figure(figsize=(12, 7))
    plt.plot(t_eval, predicted_solution, label=f'PINN Predicted Solution (c={c_val:.2f})', color='red', linestyle='-')
    plt.plot(t_eval, numerical_solution, label=f'SciPy Numerical Solution (c={c_val:.2f})', color='blue', linestyle='--')
    plt.axvline(x=c_val, color='purple', linestyle=':', label=f'IC at x=c={c_val:.2f}')
    plt.title("PINN vs. Numerical Solution")
    plt.xlabel("x (or t)")
    plt.ylabel("y(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_pinn_flux(model, num_lines):
    """Plots a flux of solutions from the PINN model."""
    print(f"\n--- Plotting PINN Solution Flux ---")
    model.eval()
    
    param_values = np.linspace(PARAM_MIN_LOADED, PARAM_MAX_LOADED, num_lines)
    t_eval = np.linspace(X_EVAL_START_LOADED, X_EVAL_END_LOADED, NUM_X_EVAL_POINTS_LOADED)
    t_tensor = torch.tensor(t_eval, dtype=torch.float32).reshape(-1, 1).to(device)
    
    plt.figure(figsize=(14, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, num_lines))

    with torch.no_grad():
        for i, c_val in enumerate(param_values):
            p_tensor = torch.tensor([[c_val]], dtype=torch.float32).expand_as(t_tensor).to(device)
            predicted_solution = model(t_tensor, p_tensor).cpu().numpy()
            plt.plot(t_eval, predicted_solution, color=colors[i], label=f'c = {c_val:.2f}')

    plt.title(f"PINN Predicted Solution Flux ({num_lines} solutions)")
    plt.xlabel("x (or t)")
    plt.ylabel("y(x)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_pinn_error_norm(model, loss_fn, trained_param_range):
    """Calculates and plots the final loss value for the trained parameter range."""
    print("\n--- Calculating and Plotting Final Loss (Error Norm) ---")
    model.eval()  # Set model to evaluation mode (good practice)
    loss_values = []

    # The 'with torch.no_grad():' block has been REMOVED from this function.
    # This is because our loss_fn itself needs to calculate gradients (derivatives)
    # to compute the physics-based loss. Gradients are required here even for evaluation.
    
    for p_val in trained_param_range:
        p_tensor = p_val.reshape(1, 1)
        # We still calculate gradients for the loss, but we won't call .backward()
        loss = loss_fn(model, p_tensor)
        loss_values.append(loss.item())
    
    plt.figure(figsize=(10, 6))
    plt.plot(trained_param_range.cpu().numpy(), loss_values, marker='o', linestyle='-')
    plt.title('Final Loss vs. Parameter Value "c"')
    plt.xlabel('Parameter "c"')
    plt.ylabel('Loss Value C(p)')
    plt.grid(True)
    plt.yscale('log') # Use log scale as loss values can vary widely
    plt.show()
def generate_data_and_save():
    """Generates ODE solution data and saves it to a .mat file."""
    print("\n--- Data Generation Setup ---")
    print(f"Using parameters loaded from '{ode_func_file_path}':")
    param_label_gen = input_param_name_in_mat.replace('_values', '')
    print(f"  Varying parameter '{param_label_gen}' from {PARAM_MIN_LOADED} to {PARAM_MAX_LOADED} ({NUM_PARAM_VALUES_LOADED} values)")
    print(f"  Fixed Initial Conditions: y(c)={GLOBAL_Y0_FIXED_AT_C}, y'(c)={GLOBAL_Y_PRIME0_FIXED_AT_C}")
    if FIXED_IC_POINT is not None:
        print(f"  Initial Conditions applied at fixed x={FIXED_IC_POINT}")
    print(f"  Solution evaluation points: x from {X_EVAL_START_LOADED} to {X_EVAL_END_LOADED} ({NUM_X_EVAL_POINTS_LOADED} points)")

    param_values_gen = np.linspace(PARAM_MIN_LOADED, PARAM_MAX_LOADED, NUM_PARAM_VALUES_LOADED).reshape(-1, 1)
    x_eval_gen = np.linspace(X_EVAL_START_LOADED, X_EVAL_END_LOADED, NUM_X_EVAL_POINTS_LOADED)
    generated_solution_vectors = np.zeros((NUM_PARAM_VALUES_LOADED, NUM_X_EVAL_POINTS_LOADED))

    print("Generating data...")
    for i in range(NUM_PARAM_VALUES_LOADED):
        current_param_val = param_values_gen[i, 0]
        initial_conditions_for_ode_solver = [GLOBAL_Y0_FIXED_AT_C, GLOBAL_Y_PRIME0_FIXED_AT_C]
        ode_func_for_scipy_gen = lambda t, y: ode_function(t, y, current_param_val)

        if input_param_name_in_mat == 'c_values':
            solver_x_start_for_ic = current_param_val
        else:
            solver_x_start_for_ic = FIXED_IC_POINT
            
        solution_vector = solve_ode_numerically(
            ode_func_for_scipy_gen, initial_conditions_for_ode_solver, 
            solver_x_start_for_ic, x_eval_gen[-1], x_eval_gen                         
        )
        generated_solution_vectors[i, :] = solution_vector
    
    print("Data generation complete.")

    generated_mat_path = f"generated_ode_data_{input_param_name_in_mat.replace('_values','')}_{NUM_PARAM_VALUES_LOADED}_pts_xe_{X_EVAL_START_LOADED}-{X_EVAL_END_LOADED}.mat"
    
    training_data = {
        input_param_name_in_mat: param_values_gen,
        'solution_vectors': generated_solution_vectors,
        'x_evaluation_points': x_eval_gen,
        'y0': GLOBAL_Y0_FIXED_AT_C,
        'y_prime0': GLOBAL_Y_PRIME0_FIXED_AT_C
    }
    if FIXED_IC_POINT is not None:
        training_data['x_start_point'] = FIXED_IC_POINT
    
    scipy.io.savemat(generated_mat_path, training_data)
    print(f"Generated data saved to '{generated_mat_path}'.")
    return generated_mat_path


def add_noise(original_mat_path, noise_std_dev_input):
    """Creates a new dataset with added white noise."""
    print(f"\n--- Adding Noise to Data from '{original_mat_path}' ---")
    try:
        original_data = scipy.io.loadmat(original_mat_path)
        solution_vectors = original_data['solution_vectors']
        x_evaluation_points = original_data['x_evaluation_points'].flatten()
        param_values = original_data[input_param_name_in_mat]
        
        noisy_solution_vectors = solution_vectors + np.random.normal(0, noise_std_dev_input, solution_vectors.shape)
        
        base_name = os.path.basename(original_mat_path)
        noisy_mat_path = base_name.replace(".mat", f"_noisy_std{noise_std_dev_input:.3f}.mat")
        
        noisy_data = {
            input_param_name_in_mat: param_values,
            'solution_vectors': noisy_solution_vectors,
            'x_evaluation_points': x_evaluation_points,
            'y0': original_data['y0'],
            'y_prime0': original_data['y_prime0']
        }
        if 'x_start_point' in original_data:
            noisy_data['x_start_point'] = original_data['x_start_point']
            
        scipy.io.savemat(noisy_mat_path, noisy_data)
        print(f"Noisy data saved to '{noisy_mat_path}'.")

        plt.figure(figsize=(12, 7))
        plt.title(f"Original vs. Noisy ODE Solutions (10 Equispaced Samples, Noise Std Dev: {noise_std_dev_input:.3f})")
        plt.xlabel("x")
        plt.ylabel("y(x)")
        plt.grid(True)
        
        total_samples = solution_vectors.shape[0]
        num_samples_to_plot = min(10, total_samples)
        sample_indices = np.unique(np.linspace(0, total_samples - 1, num_samples_to_plot, dtype=int))
        colors = plt.cm.viridis(np.linspace(0, 1, len(sample_indices)))

        for i, idx in enumerate(sample_indices):
            plt.plot(x_evaluation_points, solution_vectors[idx, :], 
                     label=f'Original {input_param_name_in_mat.replace("_values","")}={param_values[idx,0]:.2f}', 
                     linestyle='--', alpha=0.7, color=colors[i])
            plt.plot(x_evaluation_points, noisy_solution_vectors[idx, :], 
                     label=f'Noisy {input_param_name_in_mat.replace("_values","")}={param_values[idx,0]:.2f}', 
                     linestyle='-', alpha=0.7, color=colors[i])
        plt.legend()
        plt.show()

        return noisy_mat_path
    except FileNotFoundError:
        print(f"Error: Original data file '{original_mat_path}' not found for noise addition.")
        return None
    except Exception as e:
        print(f"An error occurred during noise addition: {e}")
        return None


def plot_denoiser_flux(model, x_eval_points, noisy_dataset, input_param_name, num_flux_lines=5):
    """Plots multiple NN-predicted solutions for a range of parameter values."""
    print(f"\n--- Plotting NN Output Flux (Trained on Noisy Data) for '{input_param_name.replace('_values','')}' ---")
    
    min_param = noisy_dataset.input_param_values.min().item()
    max_param = noisy_dataset.input_param_values.max().item()
    flux_param_values = np.linspace(min_param, max_param, num_flux_lines)

    plt.figure(figsize=(14, 8))
    plt.title(f"NN Predicted Solutions (Trained on Noisy Data) for varying {input_param_name.replace('_values','')}")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.grid(True)

    model.eval()
    with torch.no_grad():
        for i, param_val in enumerate(flux_param_values):
            param_tensor = torch.tensor([[param_val]], dtype=torch.float32).to(device)
            nn_predicted_solution = model(param_tensor).squeeze().cpu().numpy()
            label_prefix = f'{input_param_name.replace("_values","")}={param_val:.2f}'
            color = plt.cm.viridis(i / num_flux_lines) 
            plt.plot(x_eval_points, nn_predicted_solution, label=f'NN Predicted ({label_prefix})', linestyle='-', color=color, linewidth=2) 

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.tight_layout()
    plt.show()

def plot_error_norm(model, dataset, input_param_name, device):
    """Calculates and plots the L2 error norm for the NN model on the given dataset."""
    print(f"\n--- Calculating and Plotting Error Norm ---")
    model.eval()
    param_values = dataset.input_param_values.cpu().numpy().flatten()
    error_norms = []

    with torch.no_grad():
        for i in range(len(dataset)):
            param, true_solution = dataset[i]
            param, true_solution = param.to(device), true_solution.to(device)
            predicted_solution = model(param.unsqueeze(0)).squeeze()
            error = true_solution - predicted_solution
            error_norm = torch.norm(error).cpu().numpy()
            error_norms.append(error_norm)

    average_error_norm = np.mean(error_norms)
    print(f"Average Error Norm: {average_error_norm:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(param_values, error_norms, marker='o', linestyle='-')
    plt.title('Error Norm vs. Parameter Value')
    plt.xlabel(input_param_name.replace('_values', ''))
    plt.ylabel('Error Norm')
    plt.grid(True)
    plt.show()

    return average_error_norm


def denoiser_post_train_menu(model, denoising_criterion, denoiser_optimizer,
                             denoise_train_loader, denoise_test_loader,
                             x_eval_points_for_denoise, noisy_full_dataset, denoiser_model_path, input_param_name):
    """Presents a menu for actions after denoiser training."""
    while True:
        print("\n--- Noisy-Data Trained Model Actions ---")
        choice = get_user_choice("What do you want to do? (f)lot flux, (e)rror norm plot, (r)etrain for N epochs, (b)ack to main menu, (q)uit program? [f/e/r/b/q]: ", ['f', 'e', 'r', 'b', 'q'])

        if choice == 'f':
            num_flux_lines_str = input("Enter the number of solutions to plot for the flux (e.g., 5 or 10): ")
            try:
                num_flux_lines = int(num_flux_lines_str)
                if num_flux_lines <= 0: raise ValueError
                plot_denoiser_flux(model, x_eval_points_for_denoise, noisy_full_dataset, input_param_name, num_flux_lines)
            except ValueError:
                print("Invalid number. Please enter a positive integer.")
        elif choice == 'e':
            plot_error_norm(model, noisy_full_dataset, input_param_name, device)
        elif choice == 'r':
            try:
                epochs_to_retrain = int(input("Enter the number of additional epochs to retrain: "))
                if epochs_to_retrain <= 0: raise ValueError
                print(f"Retraining model on noisy data for {epochs_to_retrain} more epochs...")
                for epoch in range(epochs_to_retrain):
                    print(f"Retraining epoch {epoch + 1}")
                    train(denoise_train_loader, model, denoising_criterion, denoiser_optimizer, device)
                    if (epoch + 1) % 10 == 0 or epoch == epochs_to_retrain -1:
                        test(denoise_test_loader, model, denoising_criterion, device)
                save_model(model, denoiser_model_path)
                print("Retraining complete.")
            except ValueError:
                print("Invalid number of epochs. Please enter a positive integer.")
        elif choice == 'b':
            return 'back_to_main'
        elif choice == 'q':
            return 'quit_program'

def denoise_data():
    """Trains a NN on noisy data."""
    print("\n--- Training NN on Noisy Data ---")
    
    global input_param_name_in_mat 
    if not input_param_name_in_mat:
        print("Error: ODE parameters not loaded. Please load ODE function first.")
        return 'back_to_main'

    generate_noisy_choice = get_user_choice("Do you want to (g)enerate noisy data or (l)oad an existing noisy data file? [g/l]: ", ['g', 'l'])
    noisy_mat_path_actual = None
    
    if generate_noisy_choice == 'g':
        temp_original_data_file = input("Enter the path to the original (clean) .mat data file to generate noisy data from: ")
        if not os.path.exists(temp_original_data_file):
            print(f"Error: Clean data file '{temp_original_data_file}' not found. Returning.")
            return 'back_to_main'
        try:
            noise_std_to_gen = float(input("Enter the standard deviation for the white noise to generate (e.g., 0.01): "))
            if noise_std_to_gen <= 0: raise ValueError
        except ValueError:
            print("Invalid noise standard deviation. Please enter a positive number. Returning.")
            return 'back_to_main'
        noisy_mat_path_actual = add_noise(temp_original_data_file, noise_std_to_gen)
        if noisy_mat_path_actual is None:
            print("Failed to generate noisy data, cannot proceed with training.")
            return 'back_to_main'
    else: # 'l' for load
        noisy_mat_path_actual = input("Enter the path to the noisy .mat data file: ")
        if not os.path.exists(noisy_mat_path_actual):
            print(f"Error: Noisy data file '{noisy_mat_path_actual}' not found. Returning.")
            return 'back_to_main'
    
    try:
        loaded_noisy_data_for_plot = scipy.io.loadmat(noisy_mat_path_actual)
        x_eval_points_for_denoise = loaded_noisy_data_for_plot['x_evaluation_points'].flatten()
    except Exception as e:
        print(f"Warning: Could not load data for plotting. Error: {e}")

    try:
        noisy_full_dataset = ODESolutionDataset(noisy_mat_path_actual, input_param_name_in_mat)
    except Exception as e:
        print(f"Error loading noisy dataset: {e}")
        return 'back_to_main'

    denoiser_model = ODE_MLP(output_dim=len(x_eval_points_for_denoise)).to(device)
    denoiser_optimizer = torch.optim.Adam(denoiser_model.parameters())
    denoising_criterion = nn.MSELoss().to(device) 

    denoise_train_size = int(0.8 * len(noisy_full_dataset))
    denoise_test_size = len(noisy_full_dataset) - denoise_train_size
    denoise_train_dataset, denoise_test_dataset = torch.utils.data.random_split(noisy_full_dataset, [denoise_train_size, denoise_test_size])
    denoise_train_loader = DataLoader(dataset=denoise_train_dataset, batch_size=32, shuffle=True)
    denoise_test_loader = DataLoader(dataset=denoise_test_dataset, batch_size=32, shuffle=False)

    print("Training model on noisy data...")
    denoising_epochs = 200
    for epoch in range(denoising_epochs):
        train(denoise_train_loader, denoiser_model, denoising_criterion, denoiser_optimizer, device)
        if (epoch + 1) % 20 == 0:
            print(f"Noisy Training Epoch {epoch + 1}")
            test(denoise_test_loader, denoiser_model, denoising_criterion, device)
    
    match = re.search(r'_std(\d+\.\d+)\.mat$', os.path.basename(noisy_mat_path_actual))
    noise_std_dev_for_model_name = float(match.group(1)) if match else 0.01 
    denoiser_model_path = f"ode_mlp_trained_on_noisy_std{noise_std_dev_for_model_name:.3f}.pth"
    save_model(denoiser_model, denoiser_model_path)

    return denoiser_post_train_menu(denoiser_model, denoising_criterion, denoiser_optimizer,
                                     denoise_train_loader, denoise_test_loader,
                                     x_eval_points_for_denoise, noisy_full_dataset, denoiser_model_path, input_param_name_in_mat)

# --- Main Program Control Loop ---
select_ode_func = True
while select_ode_func:
    # Reset global parameters
    ode_func_file_path = ""
    input_param_name_in_mat = ""
    ode_function = None
    GLOBAL_Y0_FIXED_AT_C, GLOBAL_Y_PRIME0_FIXED_AT_C = None, None
    FIXED_IC_POINT = None
    PARAM_MIN_LOADED, PARAM_MAX_LOADED, NUM_PARAM_VALUES_LOADED = None, None, None
    X_EVAL_START_LOADED, X_EVAL_END_LOADED, NUM_X_EVAL_POINTS_LOADED = None, None, None

    # --- Dynamic ODE Function and Problem Parameters Loading ---
    ode_func_file_path = input("Enter the path to the Python file defining the ODE function (e.g., 'ode_func_eq2.py'): ")
    try:
        spec = importlib.util.spec_from_file_location("ode_module", ode_func_file_path)
        ode_module = importlib.util.module_from_spec(spec)
        sys.modules["ode_module"] = ode_module
        spec.loader.exec_module(ode_module)
        
        ode_function = ode_module.ode_system
        input_param_name_in_mat = ode_module.PARAM_NAME_IN_MAT_FILE
        print(f"Loaded ODE function from '{ode_func_file_path}' for parameter '{input_param_name_in_mat}'.")

        GLOBAL_Y0_FIXED_AT_C = ode_module.FIXED_Y0
        GLOBAL_Y_PRIME0_FIXED_AT_C = ode_module.FIXED_Y_PRIME0
        if hasattr(ode_module, 'X_START_IC_POINT'):
            FIXED_IC_POINT = ode_module.X_START_IC_POINT
        
        PARAM_MIN_LOADED = ode_module.PARAM_MIN_VALUE
        PARAM_MAX_LOADED = ode_module.PARAM_MAX_VALUE
        NUM_PARAM_VALUES_LOADED = ode_module.NUM_PARAM_VALUES_GEN
        X_EVAL_START_LOADED = ode_module.X_EVAL_START_GEN
        X_EVAL_END_LOADED = ode_module.X_EVAL_END_GEN
        NUM_X_EVAL_POINTS_LOADED = ode_module.NUM_X_EVAL_POINTS_GEN
    except Exception as e:
        print(f"Error loading ODE file: {e}. Please ensure it contains all required variables.")
        continue

    program_running = True
    while program_running:
        action_choice = get_user_choice(
            "Do you want to (p)hysics-informed training (no data), (g)enerate/load data and train, (n)oise data, (d)train NN on noisy data, (o)de function select, or (q)uit program? [p/g/n/d/o/q]: ",
            ['p', 'g', 'n', 'd', 'o', 'q']
        )
        
        if action_choice == 'p':
            menu_result_pinn = train_with_no_data()
            if menu_result_pinn == 'quit_program':
                program_running = False
                select_ode_func = False

        elif action_choice == 'g':
            generate_data_choice = input("Do you have a pre-generated data file (y/n)? ").lower()

            if generate_data_choice == 'y':
                mat_data_path = input("Enter the path to the .mat data file: ")
                if not os.path.exists(mat_data_path):
                    print(f"Error: Data file '{mat_data_path}' not found. Returning.")
                    continue 
            else:
                mat_data_path = generate_data_and_save()
                if mat_data_path is None:
                    print("Data generation failed. Returning.")
                    continue

            try:
                full_dataset = ODESolutionDataset(mat_data_path, input_param_name_in_mat)
            except Exception as e:
                print(f"Error loading dataset from '{mat_data_path}': {e}. Returning.")
                continue

            x_eval_points = full_dataset.x_evaluation_points
            y0_fixed_from_data = full_dataset.y0
            y_prime0_fixed_from_data = full_dataset.y_prime0
            x_end_plot_range = x_eval_points[-1]    

            criterion = CustomODELoss(alpha=0.0, beta=0.0, x_eval_points=x_eval_points, 
                                      y0_fixed_at_c=y0_fixed_from_data, y_prime0_fixed_at_c=y_prime0_fixed_from_data)

            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
            train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
            model = ODE_MLP(output_dim=len(x_eval_points)).to(device)
            optimizer = torch.optim.Adam(model.parameters())

            train_or_load_choice = get_user_choice("Do you want to (t)rain a new model or (l)oad a saved model? [t/l]: ", ['t', 'l'])
            model_path = f"ode_mlp_{os.path.basename(mat_data_path).replace('.mat','')}.pth"
            if train_or_load_choice == 't':
                epochs = 200
                for epoch in range(epochs):
                    print(f"Training epoch {epoch + 1}/{epochs}")
                    train(train_loader, model, criterion, optimizer, device)
                    if (epoch + 1) % 10 == 0:
                        test(test_loader, model, criterion, device)
                save_model(model, model_path)
            else: # 'l'
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    print(f"Model loaded from {model_path}")
                else:
                    print(f"No saved model found at {model_path}. Returning.")
                    continue

            print("\n--- Final Test Set Evaluation ---")
            test(test_loader, model, criterion, device)
            model.eval() 
            param_label = input_param_name_in_mat.replace('_values', '')

            def plot_single_solution():
                print(f"\n--- Plotting Predicted Solutions for user-defined '{param_label}' ---")
                while True:
                    user_param_input = input(f"Enter a value for '{param_label}' (or 'q' to quit): ")
                    if user_param_input.lower() == 'q': break
                    try:
                        param_val = float(user_param_input)
                        param_tensor = torch.tensor([[param_val]], dtype=torch.float32).to(device)
                        with torch.no_grad():
                            predicted_solution = model(param_tensor).squeeze().cpu().numpy()
                        
                        ode_func_for_scipy_plotting = lambda t, y: ode_function(t, y, param_val)
                        if input_param_name_in_mat == 'c_values':
                            solver_x_start_for_ic_plotting = param_val
                        else:
                            solver_x_start_for_ic_plotting = FIXED_IC_POINT
                        
                        scipy_solution = solve_ode_numerically(
                            ode_func_for_scipy_plotting, [y0_fixed_from_data, y_prime0_fixed_from_data],
                            solver_x_start_for_ic_plotting, x_end_plot_range, x_eval_points
                        )
                        
                        plt.figure(figsize=(12, 7))
                        plt.plot(x_eval_points, predicted_solution, label='NN Predicted Solution', color='red', linestyle='-')
                        plt.plot(x_eval_points, scipy_solution, label='SciPy Numerical Solution (validation)', color='green', linestyle=':', linewidth=2)
                        
                        if input_param_name_in_mat == 'c_values':
                            plt.axvline(x=param_val, color='gray', linestyle=':', label=f'Initial Condition Point c={param_val:.2f}')
                        else:
                            plt.axvline(x=FIXED_IC_POINT, color='gray', linestyle=':', label=f'Initial Condition Point x={FIXED_IC_POINT:.2f}')

                        plt.title(f"Solution for {param_label} = {param_val:.2f}")
                        plt.xlabel("x")
                        plt.ylabel("y(x)")
                        plt.legend()
                        plt.grid(True)
                        plt.show()
                    except ValueError:
                        print(f"Invalid input. Please enter a numerical value for '{param_label}' or 'q'.")
                    except Exception as e:
                        print(f"An error occurred during plotting: {e}")
                        print("This might be due to the ODE's behavior or choice of input value.")
                print("Exiting plotting mode.")

            def plot_flux():
                print(f"\n--- Plotting Solution Flux for '{param_label}' ---")
                try:
                    num_flux_lines = int(input("Enter the number of solutions to plot for the flux (e.g., 5 or 10): "))
                    if num_flux_lines <= 0: raise ValueError
                except ValueError:
                    print("Invalid number. Please enter a positive integer.")
                    return

                min_param = full_dataset.input_param_values.min().item()
                max_param = full_dataset.input_param_values.max().item()
                flux_param_values = np.linspace(min_param, max_param, num_flux_lines)

                plt.figure(figsize=(14, 8))
                plt.title(f"ODE Solution Flux (NN Prediction vs. SciPy) for varying {param_label}")
                plt.xlabel("x"); plt.ylabel("y(x)"); plt.grid(True)

                with torch.no_grad():
                    for i, param_val in enumerate(flux_param_values):
                        param_tensor = torch.tensor([[param_val]], dtype=torch.float32).to(device)
                        predicted_solution = model(param_tensor).squeeze().cpu().numpy()
                        
                        if input_param_name_in_mat == 'c_values':
                            solver_x_start_for_ic_plotting = param_val
                        else:
                            solver_x_start_for_ic_plotting = FIXED_IC_POINT
                        
                        ode_func_for_scipy_plotting = lambda t, y: ode_function(t, y, param_val)
                        scipy_solution = solve_ode_numerically(
                            ode_func_for_scipy_plotting, [y0_fixed_from_data, y_prime0_fixed_from_data],
                            solver_x_start_for_ic_plotting, x_end_plot_range, x_eval_points
                        )
                        
                        color = plt.cm.viridis(i / num_flux_lines)
                        plt.plot(x_eval_points, predicted_solution, label=f'NN ({param_label}={param_val:.2f})', linestyle='-', alpha=0.7, color=color)
                        plt.plot(x_eval_points, scipy_solution, label=f'SciPy ({param_label}={param_val:.2f})', linestyle='--', alpha=0.7, color=color)
                        
                        if input_param_name_in_mat == 'c_values':
                            plt.axvline(x=param_val, color=color, linestyle=':', alpha=0.5)
                
                if input_param_name_in_mat != 'c_values' and FIXED_IC_POINT is not None:
                    plt.axvline(x=FIXED_IC_POINT, color='gray', linestyle=':', label=f'Fixed IC Point x={FIXED_IC_POINT:.2f}')

                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
                plt.tight_layout(); plt.show()
                print("Exiting flux plotting mode.")

            while True:
                plot_choice = get_user_choice("Do you want to plot a (s)ingle solution, plot the (f)lux, plot (e)rror norm, or (b)ack to main menu, (q)uit? [s/f/e/b/q]: ", ['s', 'f', 'e', 'b', 'q'])
                if plot_choice == 's': plot_single_solution()
                elif plot_choice == 'f': plot_flux()
                elif plot_choice == 'e': plot_error_norm(model, full_dataset, input_param_name_in_mat, device)
                elif plot_choice == 'b': break 
                elif plot_choice == 'q':
                    program_running = False 
                    select_ode_func = False
                    break 
            
        elif action_choice == 'n':
            generate_for_noise_choice = get_user_choice("Do you want to (g)enerate the data first or (l)oad an existing clean data file to add noise to? [g/l]: ", ['g', 'l'])
            original_data_file_for_noise = None
            if generate_for_noise_choice == 'g':
                original_data_file_for_noise = generate_data_and_save()
                if original_data_file_for_noise is None:
                    print("Data generation failed, cannot add noise. Returning.")
                    continue
            else: # 'l'
                original_data_file_for_noise = input("Enter the path to the original (clean) .mat data file: ")
                if not os.path.exists(original_data_file_for_noise):
                    print(f"Error: Data file '{original_data_file_for_noise}' not found. Returning.")
                    continue
            try:
                noise_std_for_add = float(input("Enter the standard deviation for the white noise to add (e.g., 0.01): "))
                if noise_std_for_add <= 0: raise ValueError
            except ValueError:
                print("Invalid noise standard deviation. Please enter a positive number. Returning.")
                continue
            add_noise(original_data_file_for_noise, noise_std_for_add)
            continue

        elif action_choice == 'd':
            denoise_result = denoise_data()
            if denoise_result == 'quit_program':
                program_running = False 
                select_ode_func = False
            elif denoise_result == 'back_to_main':
                continue
            
        elif action_choice == 'o':
            program_running = False
            continue

        elif action_choice == 'q':
            program_running = False
            select_ode_func = False
            
    if not select_ode_func:
        break

print("\nProgram finished.")