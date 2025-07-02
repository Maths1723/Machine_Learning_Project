# main_ode_trainer.py (FINAL VERSION with corrected plot_denoiser_flux to only show NN output with varied colors and added error norm plotting)
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

# Import the new ODE-specific functions and classes
from SIMPLE_ode_support import get_device, get_user_choice, save_model, \
                               ODESolutionDataset, ODE_MLP, train, test, \
                               CustomODELoss, solve_ode_numerically

# Set environment variables
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# - - - Global Variables Initialization - - -
# These will be updated each time a new ODE function is loaded
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

# --- Helper function for Data Generation (moved for reusability) ---
def generate_data_and_save():
    """Generates ODE solution data and saves it to a .mat file.
    Returns the path to the saved file."""
    print("\n--- Data Generation Setup ---")
    print(f"Using parameters loaded from '{ode_func_file_path}':")
    param_label_gen = input_param_name_in_mat.replace('_values', '')
    print(f"  Varying parameter '{param_label_gen}' from {PARAM_MIN_LOADED} to {PARAM_MAX_LOADED} ({NUM_PARAM_VALUES_LOADED} values)")
    print(f"  Fixed Initial Conditions: y(c)={GLOBAL_Y0_FIXED_AT_C}, y'(c)={GLOBAL_Y_PRIME0_FIXED_AT_C}")
    if FIXED_IC_POINT is not None:
        print(f"  Initial Conditions applied at fixed x={FIXED_IC_POINT}")
    print(f"  Solution evaluation points: x from {X_EVAL_START_LOADED} to {X_EVAL_END_LOADED} ({NUM_X_EVAL_POINTS_LOADED} points)")

    # Generate parameter values
    param_values_gen = np.linspace(PARAM_MIN_LOADED, PARAM_MAX_LOADED, NUM_PARAM_VALUES_LOADED).reshape(-1, 1)
    x_eval_gen = np.linspace(X_EVAL_START_LOADED, X_EVAL_END_LOADED, NUM_X_EVAL_POINTS_LOADED)

    # Initialize storage
    generated_solution_vectors = np.zeros((NUM_PARAM_VALUES_LOADED, NUM_X_EVAL_POINTS_LOADED))

    print("Generating data...")
    for i in range(NUM_PARAM_VALUES_LOADED):
        current_param_val = param_values_gen[i, 0] # This is the current 'c' or 'k' value
        
        initial_conditions_for_ode_solver = [GLOBAL_Y0_FIXED_AT_C, GLOBAL_Y_PRIME0_FIXED_AT_C]
        
        ode_func_for_scipy_gen = lambda t, y: ode_function(t, y, current_param_val)

        # Determine the x_start_for_ic for solve_ode_numerically based on problem type
        if input_param_name_in_mat == 'c_values': # 'c' is the varying IC point (ode_func_eq2.py)
            solver_x_start_for_ic = current_param_val # The 'c' value itself
        else: # 'k' is the ODE parameter (ode_func_eq1.py), IC point is fixed
            solver_x_start_for_ic = FIXED_IC_POINT
            
        # Call the solve_ode_numerically function
        solution_vector = solve_ode_numerically(
            ode_func_for_scipy_gen,
            initial_conditions_for_ode_solver, 
            solver_x_start_for_ic,             
            x_eval_gen[-1],                    
            x_eval_gen                         
        )
        generated_solution_vectors[i, :] = solution_vector
    
    print("Data generation complete.")

    # Automatically set mat_data_path for the generated data
    generated_mat_path = f"generated_ode_data_{input_param_name_in_mat.replace('_values','')}_{NUM_PARAM_VALUES_LOADED}_pts_xe_{X_EVAL_START_LOADED}-{X_EVAL_END_LOADED}.mat"
    
    # Save the generated data to a .mat file
    training_data = {
        input_param_name_in_mat: param_values_gen,
        'solution_vectors': generated_solution_vectors,
        'x_evaluation_points': x_eval_gen,
        'y0': GLOBAL_Y0_FIXED_AT_C,         # Saved fixed y0 (at c or x_start_ic)
        'y_prime0': GLOBAL_Y_PRIME0_FIXED_AT_C # Saved fixed y_prime0 (at c or x_start_ic)
    }
    if FIXED_IC_POINT is not None:
        training_data['x_start_point'] = FIXED_IC_POINT
    
    scipy.io.savemat(generated_mat_path, training_data)
    print(f"Generated data saved to '{generated_mat_path}'.")
    return generated_mat_path

# --- Noise Function ---
def add_noise(original_mat_path, noise_std_dev_input):
    """
    Creates a new dataset with added white noise to the solution vectors.
    Saves the new dataset as a .mat file and returns its path.
    Also plots a sample of exactly 10 equispaced original and noisy data.
    """
    print(f"\n--- Adding Noise to Data from '{original_mat_path}' ---")
    try:
        original_data = scipy.io.loadmat(original_mat_path)
        solution_vectors = original_data['solution_vectors']
        x_evaluation_points = original_data['x_evaluation_points'].flatten()
        param_values = original_data[input_param_name_in_mat]
        
        # Create a new, noisy dataset
        noisy_solution_vectors = solution_vectors + np.random.normal(0, noise_std_dev_input, solution_vectors.shape)
        
        # Construct new file name with standard deviation
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

        # Plotting exactly 10 equispaced samples to visualize noise
        plt.figure(figsize=(12, 7))
        plt.title(f"Original vs. Noisy ODE Solutions (10 Equispaced Samples, Noise Std Dev: {noise_std_dev_input:.3f})")
        plt.xlabel("x")
        plt.ylabel("y(x)")
        plt.grid(True)
        
        # Determine the number of samples available in the dataset
        total_samples = solution_vectors.shape[0]
        
        # Select exactly 10 equispaced indices
        # Ensure that we don't try to get more samples than available
        num_samples_to_plot = min(10, total_samples)
        
        # Generate equispaced indices
        # np.linspace returns floats, so convert to integers for indexing
        sample_indices = np.unique(np.linspace(0, total_samples - 1, num_samples_to_plot, dtype=int))
        
        # Use a colormap for better visual distinction
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

# --- Denoising Plotting and Menu Functions ---
def plot_denoiser_flux(model, x_eval_points, noisy_dataset, input_param_name, num_flux_lines=5):
    """
    Plots multiple NN-predicted solutions (trained on noisy data)
    for a range of parameter values, with varied colors.
    """
    print(f"\n--- Plotting NN Output Flux (Trained on Noisy Data) for '{input_param_name.replace('_values','')}' ---")
    
    min_param = noisy_dataset.input_param_values.min().item()
    max_param = noisy_dataset.input_param_values.max().item()
    flux_param_values = np.linspace(min_param, max_param, num_flux_lines)

    plt.figure(figsize=(14, 8))
    plt.title(f"NN Predicted Solutions (Trained on Noisy Data) for varying {input_param_name.replace('_values','')}") # Updated title
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.grid(True)

    model.eval()
    with torch.no_grad():
        for i, param_val in enumerate(flux_param_values):
            param_tensor = torch.tensor([[param_val]], dtype=torch.float32).to(device)
            nn_predicted_solution = model(param_tensor).squeeze().cpu().numpy()

            label_prefix = f'{input_param_name.replace("_values","")}={param_val:.2f}'
            # Use colormap for shading of NN predicted output
            color = plt.cm.viridis(i / num_flux_lines) 

            plt.plot(x_eval_points, nn_predicted_solution, label=f'NN Predicted ({label_prefix})', linestyle='-', color=color, linewidth=2) 

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.tight_layout()
    plt.show()

def plot_error_norm(model, dataset, input_param_name, device):
    """
    Calculates and plots the average error norm for the NN model on the given dataset.
    """
    print(f"\n--- Calculating and Plotting Error Norm ---")
    model.eval()
    param_values = dataset.input_param_values.cpu().numpy().flatten()
    error_norms = []

    with torch.no_grad():
        for i in range(len(dataset)):
            param = dataset.input_param_values[i].to(device)
            # For error norm, we compare to the 'true' (noisy or clean) solution from the dataset
            # If this function is called after denoising, dataset is noisy_full_dataset, 
            # so true_solution is the noisy solution.
            # If called after main training, dataset is full_dataset, so true_solution is the clean numerical solution.
            true_solution = dataset.solution_vectors[i].to(device) 
            predicted_solution = model(param).squeeze()
            error = true_solution - predicted_solution
            error_norm = torch.norm(error).cpu().numpy() # L2 norm of the error vector
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
        print("\n--- Noisy-Data Trained Model Actions ---") # Updated menu title
        choice = get_user_choice("What do you want to do? (f)lot flux, (e)rror norm plot, (r)etrain for N epochs, (b)ack to main menu, (q)uit program? [f/e/r/b/q]: ", ['f', 'e', 'r', 'b', 'q'])

        if choice == 'f': # Changed from 'p' to 'f'
            num_flux_lines_str = input("Enter the number of solutions to plot for the flux (e.g., 5 or 10): ")
            try:
                num_flux_lines = int(num_flux_lines_str)
                if num_flux_lines <= 0:
                    raise ValueError
                # Pass noisy_full_dataset as the source of solutions
                plot_denoiser_flux(model, x_eval_points_for_denoise, noisy_full_dataset, input_param_name, num_flux_lines)
            except ValueError:
                print("Invalid number. Please enter a positive integer.")

        elif choice == 'e': # New: Error norm plot
            plot_error_norm(model, noisy_full_dataset, input_param_name, device)

        elif choice == 'r':
            epochs_to_retrain_str = input("Enter the number of additional epochs to retrain: ")
            try:
                epochs_to_retrain = int(epochs_to_retrain_str)
                if epochs_to_retrain <= 0:
                    raise ValueError
                print(f"Retraining model on noisy data for {epochs_to_retrain} more epochs...")
                for epoch in range(epochs_to_retrain):
                    print(f"Retraining epoch {epoch + 1}")
                    train(denoise_train_loader, model, denoising_criterion, denoiser_optimizer, device)
                    if (epoch + 1) % 10 == 0 or epoch == epochs_to_retrain -1: # Test every 10 epochs or on last epoch
                        test(denoise_test_loader, model, denoising_criterion, device)
                save_model(model, denoiser_model_path) # Save after retraining
                print("Retraining complete.")
            except ValueError:
                print("Invalid number of epochs. Please enter a positive integer.")
            
        elif choice == 'b':
            return 'back_to_main' # Signal to return to the main program loop
            
        elif choice == 'q':
            return 'quit_program' # Signal to quit the entire program

def denoise_data(): # Removed original_mat_path parameter
    """
    Trains a NN on noisy data.
    It handles generating noisy data if not found.
    Saves the trained model, then offers post-training actions.
    """
    print("\n--- Training NN on Noisy Data ---")
    
    global input_param_name_in_mat 
    if not input_param_name_in_mat:
        print("Error: ODE parameters not loaded. Please load ODE function first.")
        return 'back_to_main'

    noisy_mat_path_actual = None 

    generate_noisy_choice = get_user_choice("Do you want to (g)enerate noisy data or (l)oad an existing noisy data file? [g/l]: ", ['g', 'l'])

    if generate_noisy_choice == 'g':
        # Need a clean data file to add noise to, usually the one generated by 'g' option.
        # For simplicity, if not available, assume it was generated in the 'g' menu.
        # If user chooses 'g' here, they must have gone through the 'g' menu first.
        temp_original_data_file = input("Enter the path to the original (clean) .mat data file to generate noisy data from: ")
        if not os.path.exists(temp_original_data_file):
            print(f"Error: Clean data file '{temp_original_data_file}' not found. Cannot generate noisy data. Returning to main menu.")
            return 'back_to_main'

        try:
            noise_std_to_gen = float(input("Enter the standard deviation for the white noise to generate (e.g., 0.01): "))
            if noise_std_to_gen <= 0:
                raise ValueError
        except ValueError:
            print("Invalid noise standard deviation. Please enter a positive number. Returning to main menu.")
            return 'back_to_main'
        noisy_mat_path_actual = add_noise(temp_original_data_file, noise_std_to_gen)
        if noisy_mat_path_actual is None:
            print("Failed to generate noisy data, cannot proceed with training.")
            return 'back_to_main'
    else: # 'l' for load existing noisy file
        noisy_mat_path_actual = input("Enter the path to the noisy .mat data file: ")
        if not os.path.exists(noisy_mat_path_actual):
            print(f"Error: Noisy data file '{noisy_mat_path_actual}' not found. Returning to main menu.")
            return 'back_to_main'
    
    # --- Plot the noisy data alone (Existing Request) ---
    try:
        loaded_noisy_data_for_plot = scipy.io.loadmat(noisy_mat_path_actual)
        noisy_solutions_for_plot = loaded_noisy_data_for_plot['solution_vectors']
        x_eval_points_for_denoise = loaded_noisy_data_for_plot['x_evaluation_points'].flatten()
        param_values_for_plot = loaded_noisy_data_for_plot[input_param_name_in_mat]

        plt.figure(figsize=(12, 7))
        plt.title("Sample of Noisy ODE Solutions (Input to NN Training)")
        plt.xlabel("x")
        plt.ylabel("y(x)")
        plt.grid(True)
        
        num_samples_to_plot_noisy = min(5, noisy_solutions_for_plot.shape[0])
        sample_indices_noisy = np.random.choice(noisy_solutions_for_plot.shape[0], num_samples_to_plot_noisy, replace=False)

        for i, idx in enumerate(sample_indices_noisy):
            plt.plot(x_eval_points_for_denoise, noisy_solutions_for_plot[idx, :], 
                             label=f'Noisy {input_param_name_in_mat.replace("_values","")}={param_values_for_plot[idx,0]:.2f}', 
                             linestyle='-', alpha=0.7, color=plt.cm.viridis(i / num_samples_to_plot_noisy))
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Warning: Could not plot noisy data sample. Error: {e}")


    # Load the noisy dataset. This will be both input (param) and target (noisy solution)
    try:
        noisy_full_dataset = ODESolutionDataset(noisy_mat_path_actual, input_param_name_in_mat)
    except Exception as e:
        print(f"Error loading noisy dataset: {e}")
        return 'back_to_main'

    # The NN will learn to map parameter -> noisy_solution
    denoiser_model = ODE_MLP(output_dim=len(x_eval_points_for_denoise)).to(device)
    denoiser_optimizer = torch.optim.Adam(denoiser_model.parameters())
    
    # Criterion will compare predicted solution with the noisy solution
    denoising_criterion = nn.MSELoss().to(device) 

    # Split for training the model on noisy data
    denoise_train_size = int(0.8 * len(noisy_full_dataset))
    denoise_test_size = len(noisy_full_dataset) - denoise_train_size
    denoise_train_dataset, denoise_test_dataset = torch.utils.data.random_split(noisy_full_dataset, [denoise_train_size, denoise_test_size])

    denoise_train_loader = DataLoader(dataset=denoise_train_dataset, batch_size=32, shuffle=True)
    denoise_test_loader = DataLoader(dataset=denoise_test_dataset, batch_size=32, shuffle=False)

    print("Training model on noisy data...")
    denoising_epochs = 200 # Can be adjusted
    for epoch in range(denoising_epochs):
        denoiser_model.train()
        total_loss = 0
        for param_batch, noisy_solution_batch in denoise_train_loader: # Target is now noisy_solution_batch
            param_batch, noisy_solution_batch = param_batch.to(device), noisy_solution_batch.to(device)

            denoiser_optimizer.zero_grad()
            predicted_solution = denoiser_model(param_batch)
            loss = denoising_criterion(predicted_solution, noisy_solution_batch) # Compare to noisy target
            loss.backward()
            denoiser_optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 20 == 0:
            print(f"Noisy Training Epoch {epoch + 1}, Loss: {total_loss / len(denoise_train_loader):.4f}")

            denoiser_model.eval()
            test_loss = 0
            with torch.no_grad():
                for param_batch, noisy_solution_batch in denoise_test_loader: # Test against noisy target
                    param_batch, noisy_solution_batch = param_batch.to(device), noisy_solution_batch.to(device)
                    predicted_solution = denoiser_model(param_batch)
                    loss = denoising_criterion(predicted_solution, noisy_solution_batch)
                    test_loss += loss.item()
            print(f"Noisy Training Test Loss: {test_loss / len(denoise_test_loader):.4f}")
    
    # Extract noise_std_dev from filename for model saving, or use default
    base_name_noisy = os.path.basename(noisy_mat_path_actual)
    match = re.search(r'_std(\d+\.\d+)\.mat$', base_name_noisy)
    noise_std_dev_for_model_name = float(match.group(1)) if match else 0.01 
    
    denoiser_model_path = f"ode_mlp_trained_on_noisy_{base_name_noisy.replace('.mat', '')}_std{noise_std_dev_for_model_name:.3f}.pth"
    save_model(denoiser_model, denoiser_model_path)
    print(f"Model trained on noisy data saved to '{denoiser_model_path}'.")

    # Pass noisy_full_dataset as the reference for plotting (it contains the noisy solutions)
    menu_result = denoiser_post_train_menu(denoiser_model, denoising_criterion, denoiser_optimizer,
                                             denoise_train_loader, denoise_test_loader,
                                             x_eval_points_for_denoise, noisy_full_dataset, denoiser_model_path, input_param_name_in_mat)
    return menu_result 

# --- Main Program Control Loop ---
select_ode_func = True
while select_ode_func:
    # Reset global parameters for each ODE function selection
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

    # --- Dynamic ODE Function and Problem Parameters Loading ---
    ode_func_file_path = input("Enter the path to the Python file defining the ODE function (e.g., 'ode_func_eq1.py' or 'ode_func_eq2.py' or 'ode_func_eq3.py'): ")

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
        else:
            FIXED_IC_POINT = None 

        PARAM_MIN_LOADED = ode_module.PARAM_MIN_VALUE
        PARAM_MAX_LOADED = ode_module.PARAM_MAX_VALUE
        NUM_PARAM_VALUES_LOADED = ode_module.NUM_PARAM_VALUES_GEN
        X_EVAL_START_LOADED = ode_module.X_EVAL_START_GEN
        X_EVAL_END_LOADED = ode_module.X_EVAL_END_GEN
        NUM_X_EVAL_POINTS_LOADED = ode_module.NUM_X_EVAL_POINTS_GEN

    except FileNotFoundError:
        print(f"Error: ODE function file '{ode_func_file_path}' not found. Returning to ODE selection.")
        continue # Go back to the start of the outer loop to select ODE function again
    except AttributeError as e:
        print(f"Error: Make sure '{ode_func_file_path}' defines 'ode_system', 'PARAM_NAME_IN_MAT_FILE', and all required problem parameters (e.g., FIXED_Y0, PARAM_MIN_VALUE, PARAM_MAX_VALUE, NUM_PARAM_VALUES_GEN, X_EVAL_START_GEN, X_EVAL_END_GEN, NUM_X_EVAL_POINTS_GEN).")
        print(f"Missing attribute: {e}. Returning to ODE selection.")
        continue # Go back to the start of the outer loop
    except Exception as e:
        print(f"An unexpected error occurred while loading ODE function or parameters: {e}. Returning to ODE selection.")
        continue # Go back to the start of the outer loop

    # --- Main Program Flow (g/n/d menu) ---
    program_running = True
    while program_running:
        action_choice = get_user_choice("Do you want to (g)enerate/load data and train, (n)oise data, (d)train NN on noisy data, (o)de function select, or (q)uit program? [g/n/d/o/q]: ", ['g', 'n', 'd', 'o', 'q'])

        if action_choice == 'g':
            generate_data_choice = input("Do you have a pre-generated data file (y/n)? ").lower()

            # --- Data Loading or Generation Logic ---
            if generate_data_choice == 'y':
                mat_data_path = input("Enter the path to the .mat data file: ")
                if not os.path.exists(mat_data_path):
                    print(f"Error: Data file '{mat_data_path}' not found. Returning to main menu.")
                    continue 
            else: # User wants to generate data
                mat_data_path = generate_data_and_save()
                if mat_data_path is None:
                    print("Data generation failed. Returning to main menu.")
                    continue

            base_name = os.path.basename(mat_data_path)
            model_name_prefix = base_name.replace(".mat", "")
            model_path = f"ode_mlp_{model_name_prefix}.pth"

            epochs = 200   
            batch_size = 32
            optimizer_class = torch.optim.Adam
            alpha_coeff = 0.0     #just for testing, reupdate
            beta_coeff = 0.0

            try:
                full_dataset = ODESolutionDataset(mat_data_path, input_param_name_in_mat)
            except FileNotFoundError:
                print(f"Error: The data file '{mat_data_path}' was not found. Returning to main menu.")
                continue
            except KeyError:
                print(f"Error: The parameter '{input_param_name_in_mat}' not found in '{mat_data_path}'. Returning to main menu.")
                continue


            x_eval_points = full_dataset.x_evaluation_points
            y0_fixed_from_data = full_dataset.y0
            y_prime0_fixed_from_data = full_dataset.y_prime0
            x_start_plot_range = x_eval_points[0]    
            x_end_plot_range = x_eval_points[-1]    

            criterion = CustomODELoss(alpha=alpha_coeff, beta=beta_coeff,
                                      x_eval_points=x_eval_points,
                                      y0_fixed_at_c=y0_fixed_from_data,    
                                      y_prime0_fixed_at_c=y_prime0_fixed_from_data).to(device)

            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

            model = ODE_MLP(output_dim=len(x_eval_points)).to(device)
            optimizer = optimizer_class(model.parameters())

            train_or_load_choice = get_user_choice("Do you want to (t)rain a new model or (l)oad a saved model? [t/l]: ", ['t', 'l'])
            if train_or_load_choice == 't':
                for epoch in range(epochs):
                    print(f"Training epoch {epoch + 1}")
                    train(train_loader, model, criterion, optimizer, device)
                    if (epoch + 1) % 10 == 0:
                        print(f"--- Epoch {epoch + 1} Test ---")
                        test(test_loader, model, criterion, device)
                save_model(model, model_path)
                
            elif train_or_load_choice == 'l':
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    print(f"Model loaded from {model_path}")
                else:
                    print(f"No saved model found at {model_path}. Returning to main menu.")
                    continue

            print("\n--- Final Test Set Evaluation ---")
            test(test_loader, model, criterion, device)

            model.eval() 
            param_label = input_param_name_in_mat.replace('_values', '')

            def plot_single_solution():
                """Plots the predicted and true solution for a single user-defined parameter value."""
                print(f"\n--- Plotting Predicted Solutions for user-defined '{param_label}' ---")
                while True:
                    try:
                        user_param_input = input(f"Enter a value for '{param_label}' (or 'q' to quit): ")
                        if user_param_input.lower() == 'q':
                            break
                            
                        param_val = float(user_param_input)
                            
                        min_param = full_dataset.input_param_values.min().item()
                        max_param = full_dataset.input_param_values.max().item()

                        if not (min_param <= param_val <= max_param):
                            print(f"Warning: '{param_label}' value is outside the training range [{min_param:.2f}, {max_param:.2f}]. Prediction might be less accurate.")

                        param_tensor = torch.tensor([[param_val]], dtype=torch.float32).to(device)
                            
                        with torch.no_grad():
                            predicted_solution_tensor = model(param_tensor)
                            predicted_solution = predicted_solution_tensor.squeeze().cpu().numpy()
                            
                        closest_param_idx = torch.argmin(torch.abs(full_dataset.input_param_values - param_val)).item()
                        true_numerical_solution_from_dataset = full_dataset.solution_vectors[closest_param_idx].cpu().numpy()

                        initial_conditions_for_plotting_ode = [y0_fixed_from_data, y_prime0_fixed_from_data]
                        
                        if input_param_name_in_mat == 'c_values':
                            solver_x_start_for_ic_plotting = param_val
                        else:
                            solver_x_start_for_ic_plotting = FIXED_IC_POINT
                            
                        ode_func_for_scipy_plotting = lambda t, y: ode_function(t, y, param_val)
                        scipy_solution = solve_ode_numerically(
                            ode_func_for_scipy_plotting,
                            initial_conditions_for_plotting_ode, 
                            solver_x_start_for_ic_plotting,             
                            x_end_plot_range,                    
                            x_eval_points                          
                        )
                            
                        plt.figure(figsize=(12, 7))
                        plt.plot(x_eval_points, true_numerical_solution_from_dataset, label='Closest parameter Solution (from dataset)', color='blue', linestyle='--', linewidth=2)
                        plt.plot(x_eval_points, predicted_solution, label='NN Predicted Solution', color='red', marker='o', markersize=3, linestyle='-')
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
                """Plots multiple predicted and numerical solutions for a range of parameter values."""
                print(f"\n--- Plotting Solution Flux for '{param_label}' ---")
                num_flux_lines_str = input("Enter the number of solutions to plot for the flux (e.g., 5 or 10): ")
                try:
                    num_flux_lines = int(num_flux_lines_str)
                    if num_flux_lines <= 0:
                        raise ValueError
                except ValueError:
                    print("Invalid number. Please enter a positive integer.")
                    return

                min_param = full_dataset.input_param_values.min().item()
                max_param = full_dataset.input_param_values.max().item()
                flux_param_values = np.linspace(min_param, max_param, num_flux_lines)

                plt.figure(figsize=(14, 8))
                plt.title(f"ODE Solution Flux (NN Prediction vs. SciPy) for varying {param_label}")
                plt.xlabel("x")
                plt.ylabel("y(x)")
                plt.grid(True)

                with torch.no_grad():
                    for i, param_val in enumerate(flux_param_values):
                        param_tensor = torch.tensor([[param_val]], dtype=torch.float32).to(device)
                        predicted_solution_tensor = model(param_tensor)
                        predicted_solution = predicted_solution_tensor.squeeze().cpu().numpy()

                        initial_conditions_for_plotting_ode = [y0_fixed_from_data, y_prime0_fixed_from_data]
                        if input_param_name_in_mat == 'c_values':
                            solver_x_start_for_ic_plotting = param_val
                        else:
                            solver_x_start_for_ic_plotting = FIXED_IC_POINT
                            
                        ode_func_for_scipy_plotting = lambda t, y: ode_function(t, y, param_val)
                        scipy_solution = solve_ode_numerically(
                            ode_func_for_scipy_plotting,
                            initial_conditions_for_plotting_ode, 
                            solver_x_start_for_ic_plotting,             
                            x_end_plot_range,                    
                            x_eval_points                          
                        )
                        
                        plt.plot(x_eval_points, predicted_solution, label=f'NN ({param_label}={param_val:.2f})', linestyle='-', alpha=0.7, color=plt.cm.viridis(i / num_flux_lines))
                        plt.plot(x_eval_points, scipy_solution, label=f'SciPy ({param_label}={param_val:.2f})', linestyle='--', alpha=0.7, color=plt.cm.viridis(i / num_flux_lines))
                        
                        if input_param_name_in_mat == 'c_values':
                            plt.axvline(x=param_val, color=plt.cm.viridis(i / num_flux_lines), linestyle=':', alpha=0.5)

                if input_param_name_in_mat != 'c_values' and FIXED_IC_POINT is not None:
                    plt.axvline(x=FIXED_IC_POINT, color='gray', linestyle=':', label=f'Fixed IC Point x={FIXED_IC_POINT:.2f}')

                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
                plt.tight_layout()
                plt.show()
                print("Exiting flux plotting mode.")

            # --- Main Plotting Menu for ODE Solution NN ---
            while True:
                # Added 'e' for error norm plot option
                plot_choice = get_user_choice("Do you want to plot a (s)ingle solution, plot the (f)lux, plot (e)rror norm, or (b)ack to main menu, (q)uit? [s/f/e/b/q]: ", ['s', 'f', 'e', 'b', 'q'])
                if plot_choice == 's':
                    plot_single_solution()
                elif plot_choice == 'f':
                    plot_flux()
                elif plot_choice == 'e': # New: Call error norm plot
                    plot_error_norm(model, full_dataset, input_param_name_in_mat, device)
                elif plot_choice == 'b':
                    break 
                elif plot_choice == 'q':
                    program_running = False 
                    select_ode_func = False # Signal to exit outer loop
                    break 
            
        elif action_choice == 'n':
            # This branch now calls add_noise which asks for SD, and then returns to main menu
            # No additional prompt for SD here as add_noise handles it.
            generate_for_noise_choice = get_user_choice("Do you want to (g)enerate the data first or (l)oad an existing clean data file to add noise to? [g/l]: ", ['g', 'l'])
            
            original_data_file_for_noise = None
            if generate_for_noise_choice == 'g':
                original_data_file_for_noise = generate_data_and_save()
                if original_data_file_for_noise is None:
                    print("Data generation failed, cannot proceed with noise addition. Returning to main menu.")
                    continue
            else: # 'l' for load
                original_data_file_for_noise = input("Enter the path to the original (clean) .mat data file to add noise to: ")
                if not os.path.exists(original_data_file_for_noise):
                    print(f"Error: Data file '{original_data_file_for_noise}' not found. Returning to main menu.")
                    continue
            
            try:
                noise_std_for_add = float(input("Enter the standard deviation for the white noise to add (e.g., 0.01): "))
                if noise_std_for_add <= 0:
                    raise ValueError
            except ValueError:
                print("Invalid noise standard deviation. Please enter a positive number. Returning to main menu.")
                continue

            add_noise(original_data_file_for_noise, noise_std_for_add)
            continue # Return to main g/n/d menu after noising

        elif action_choice == 'd':
            # denoise_data now only takes noisy data path and trains NN on it
            denoise_result = denoise_data()
            if denoise_result == 'quit_program':
                program_running = False 
                select_ode_func = False # Signal to exit outer loop
            elif denoise_result == 'back_to_main':
                continue # Go back to the main menu
            
        elif action_choice == 'o': # Option to select a new ODE function
            program_running = False # Break out of inner loop
            continue # Continue outer loop (to re-prompt for ODE function)

        elif action_choice == 'q': # Option to quit program
            program_running = False # Break out of inner loop
            select_ode_func = False # Signal to exit outer loop
            
    if not select_ode_func: # Check if a quit signal was set from any menu
        break # Exit the outermost while loop

print("Program finished.")