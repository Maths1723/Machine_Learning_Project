Markdown

# Neural Network Based Ordinary Differential Equation Solver

## Project Overview

This repository presents a versatile framework for training a neural network to predict solutions to Ordinary Differential Equations (ODEs). Both as PINN and as hybrid numerical NNs methods. Rather than solving each ODE instance from scratch, the neural network learns a mapping from a varying ODE parameter (e.g., a coefficient or an initial condition point) directly to the corresponding solution curve. This approach leverages the power of machine learning to approximate the solution manifold of a family of ODEs, enabling rapid inference once trained.

The framework integrates `scipy.integrate.solve_ivp` for high-fidelity numerical data generation and validation, ensuring that the neural network's predictions are benchmarked against reliable mathematical ground truth. A custom loss function combines data fidelity with the enforcement of boundary conditions, guiding the network towards physically consistent solutions.

## Key Features

* **Dynamic ODE Loading:** Easily switch between different ODE definitions by specifying a Python file at runtime.
* **Automated Data Generation:** Generate high-accuracy training data for families of ODEs using SciPy's robust numerical integrators.
* **Flexible Parameterization:** Supports ODEs where either:
    * An ODE coefficient varies, with fixed initial conditions at a specific point.
    * The point at which initial conditions are applied varies, with fixed initial values.
* **Custom Neural Network Architecture:** A simple yet effective Multi-Layer Perceptron (MLP) capable of learning complex solution mappings.
* **Physics-Informed Loss Function:** A custom loss that combines:
    * Mean Squared Error (MSE) for fidelity to numerical solutions.
    * Boundary condition enforcement terms to ensure learned solutions satisfy initial values.
* **GPU Acceleration:** Utilizes PyTorch for efficient training on CUDA or MPS-enabled devices.
* **Interactive Visualization:** Real-time plotting of predicted, dataset-derived, and fresh SciPy-generated solutions for intuitive comparison and qualitative assessment of generalization.
* **Model Persistence:** Save and load trained neural network models for continued use or deployment.

## Mathematical and Theoretical Background

Ordinary Differential Equations (ODEs) are fundamental tools in modeling dynamic systems across science and engineering. While analytical solutions are elegant, they are often unattainable for all but the simplest ODEs. Numerical methods, such as Runge-Kutta methods (e.g., RK45 employed by `solve_ivp`), provide accurate approximations.

This project delves into a machine learning paradigm for solving ODEs that transcends single-instance numerical solutions. Instead of computing $y(x)$ for one $c$, we train a function approximator $f(c) \to y(x)$ where $y(x)$ is a vector representing the solution at discrete evaluation points. This $f$ is our neural network.

The core idea is to learn the **solution manifold** – the high-dimensional space of all possible solutions as a function of the input parameter $c$. By training on a diverse set of $(c, y(x))$ pairs, the neural network learns to implicitly capture the underlying differential relationships.

Our custom `CustomODELoss` embodies a "Physics-Informed" philosophy, albeit in a data-driven context (distinguished from a pure Physics-Informed Neural Network (PINN) which would directly incorporate the ODE residual into the loss for continuous domains, as instead done in nodatatrain). Here, the "physics" is enforced through two primary components:

1.  **Data Fidelity Term (Integral Term):** This is a standard Mean Squared Error (MSE) between the neural network's predicted solution curve $y_{pred}(x)$ and the target solution curve $y_{true}(x)$ (obtained from a high-accuracy numerical solver). This term drives the network to match the general shape and values of the known solutions.
    $L_{data} = \frac{1}{N} \sum_{i=1}^{N} \left( y_{pred}(x_i) - y_{true}(x_i) \right)^2$

2.  **Boundary Condition (BC) Enforcement Terms:** These terms act as a form of regularization, ensuring that the initial conditions ($y(c)=y_0$ and $y'(c)=y'_0$) are satisfied by the neural network's predicted solution at the corresponding initial point $c$.
    * $L_{y_0} = \alpha \left( y_{pred}(c) - y_0 \right)^2$
    * $L_{y'_0} = \beta \left( y'_{pred}(c) - y'_0 \right)^2$
    Here, $y'_{pred}(c)$ is approximated via numerical differentiation of the neural network's output. The coefficients $\alpha$ and $\beta$ control the importance of these boundary conditions relative to the data fidelity term.

The total loss minimized during training is:
$L_{total} = L_{data} + L_{y_0} + L_{y'_0}$

This hybrid loss steers the neural network to not only learn the solution curves accurately but also to adhere to the fundamental initial conditions, which are critical physical constraints of the ODE system.

## Project Structure

.
├── main_ode_trainer.py         # Main script for data generation, training, and evaluation.
├── SIMPLE_ode_support.py       # Core utilities: NN model, CustomLoss, Dataset, ODE solver wrapper.
├── ode_func_files


## Setup and Installation

1.  **Clone the Repository:**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**

    ```bash
    pip install torch numpy scipy matplotlib
    ```
    * `torch`: PyTorch for neural network operations and GPU acceleration.
    * `numpy`: Numerical operations, especially for data handling.
    * `scipy`: Crucial for `scipy.integrate.solve_ivp` (numerical ODE solver) and `.mat` file operations.
    * `matplotlib`: For plotting and visualization.

## How to Use the Project

The `main_ode_trainer.py` script orchestrates the entire workflow.

### 1. Running the Trainer

Execute the main script from your terminal:

```bash
python main_ode_trainer.py
You will be prompted for several inputs:

a. ODE Function File Path:
* **Prompt:** `Enter the path to the Python file defining the ODE function (e.g., 'ode_func_eq1.py' or 'ode_func_eq2.py'):`
* **Input:** Provide the relative or absolute path to one of your ODE definition files, e.g., `ode_func_eq1.py` or `ode_func_eq2.py`.
* **Explanation:** This dynamically loads the specific ODE system and its associated parameters (varying parameter range, fixed initial conditions, evaluation points) into the trainer.
b. Data Source:
* **Prompt:** `Do you have a pre-generated data file (y/n)?`
* **Input:**
    * `y`: If you have previously generated data and saved it as a `.mat` file. You will then be prompted to enter the path to this `.mat` file. This saves computation time for subsequent runs.
    * `n`: To generate new data. The script will use the parameters defined within the chosen ODE function file (e.g., `ode_func_eq1.py`) to numerically solve the ODE for a range of parameter values and save them as a new `.mat` file.
* **Explanation:** Data generation can be computationally intensive, especially for many parameter values or fine evaluation grids. Using pre-generated data allows for faster iteration on model training.
c. Training or Loading a Model:
* **Prompt:** `Do you want to (t)rain a new model or (l)oad a saved model? [t/l]:`
* **Input:**
    * `t`: To train a new neural network model from scratch using the loaded or generated data. The training process will display loss values per batch.
    * `l`: To load a previously trained model (e.g., from an `.pth` file). This is useful for evaluating an already trained model or continuing training from a checkpoint.
* **Explanation:** This choice allows for flexibility in your workflow, whether you're developing a new model or evaluating an existing one.
2. Understanding Training Output
During training (t option), you'll see output similar to:

Training epoch 1
loss: 0.123456  [  32/ 800]
loss: 0.087654  [ 640/ 800]
--- Epoch 10 Test ---
Test Error: 
 Avg loss: 0.001234 
loss: The CustomODELoss value for the current batch.
[current/size]: Progress within the current epoch (current samples processed / total training samples).
Test Error: The average loss calculated on the separate test dataset after every 10 epochs. This indicates how well the model generalizes to unseen data within the same distribution as the training data.
3. Interactive Plotting and Evaluation
After training or loading a model, the script enters an interactive plotting loop:

Prompt: Enter a value for 'c' (or 'q' to quit): (The parameter label, e.g., 'c' or 'k', will adapt based on PARAM_NAME_IN_MAT_FILE.)
Input: Enter a numerical value for the varying parameter (e.g., 0.5, 1.2, 0.05) or q to quit.
For each numerical input, a plot will be displayed showing three distinct curves:

NN Predicted Solution (Red, Markers): This is the output of your trained neural network for the $c$ value you entered. It's the model's learned approximation.
True Solution (from dataset) (Blue, Dashed): This curve represents the actual numerical solution from your training/test dataset for the $c$ value closest to your input $c$. This serves as the supervised learning target.
SciPy Numerical Solution (validation) (Green, Dotted): This is a freshly computed numerical solution for the exact $c$ value you entered, calculated using scipy.integrate.solve_ivp. This is your independent, high-fidelity ground truth for validation.
Important Considerations during Plotting:

Interpolation vs. Extrapolation:
If your input $c$ is within the PARAM_MIN_VALUE and PARAM_MAX_VALUE range defined in your ODE function file, expect the "NN Predicted Solution" to closely match both the "True Solution (from dataset)" (if one exists very close by) and the "SciPy Numerical Solution (validation)". This demonstrates successful interpolation.
If your input $c$ is outside this training range, the script will issue a Warning: 'c' value is outside the training range [...]. In this case, the "NN Predicted Solution" might deviate significantly from the "SciPy Numerical Solution (validation)". This illustrates the limitations of neural networks in extrapolation. The "True Solution (from dataset)" will represent the data point closest to the training boundary, not the extrapolated point itself.
Adding New ODE Function Files
The project is designed to be extensible. To train a model for a new ODE:

Create a New Python File: Create a new .py file (e.g., ode_func_new.py) in the same directory as main_ode_trainer.py.

Define Required Parameters: Inside this file, define all the required problem-specific parameters. Crucially, adhere strictly to the naming conventions:

PARAM_NAME_IN_MAT_FILE (e.g., 'alpha_values')
PARAM_MIN_VALUE
PARAM_MAX_VALUE
NUM_PARAM_VALUES_GEN
FIXED_Y0
FIXED_Y_PRIME0
X_EVAL_START_GEN
X_EVAL_END_GEN
NUM_X_EVAL_POINTS_GEN
(Conditional) X_START_IC_POINT: Only define this if the initial conditions are fixed at a specific $x$ coordinate. Do NOT define it if the varying parameter (e.g., $c$) is the initial condition point itself.
(Optional) Fixed ODE Coefficients: Define any other fixed coefficients used in your ODE (e.g., FIXED_GRAVITY = 9.81).
Implement the ode_system Function:

The function signature must be ode_system(x, y, param_val).
$x$ is the independent variable.
$y$ is a list/array $[y, y']$.
param_val is the current value of the parameter defined by PARAM_NAME_IN_MAT_FILE.
Return a list [dy/dx, d2y/dx2]. For a second-order ODE, dy/dx will always be y[1].
Handle Singularities: If your ODE involves division by $x$ or other terms that become undefined at certain points, include if checks within ode_system to return np.inf or np.nan (though it's best practice to set X_EVAL_START_GEN and PARAM_MIN_VALUE to avoid these points directly during data generation/evaluation).
Run main_ode_trainer.py: When prompted, provide the path to your new ODE function file.

Customization and Hyperparameters
You can customize the training process and neural network architecture by modifying main_ode_trainer.py and SIMPLE_ode_support.py:

In main_ode_trainer.py:
epochs: Number of full passes through the training dataset. More epochs can lead to better learning but also overfitting.
batch_size: Number of samples processed in each training step. Larger batches might lead to more stable gradients but slower updates.
optimizer_class: Change to torch.optim.SGD, torch.optim.AdamW, etc.
alpha_coeff, beta_coeff: Coefficients for the boundary condition terms in CustomODELoss. Adjust these to control the emphasis on satisfying initial conditions. Higher values enforce BCs more strictly but might impact overall data fit.
In SIMPLE_ode_support.py:
ODE_MLP class:
input_dim: Default is 1 (for the single varying parameter $c$).
output_dim: Must match NUM_X_EVAL_POINTS_GEN from your ODE function file.
hidden_layers: Number of hidden layers in the MLP.
hidden_dim: Number of neurons in each hidden layer.
Experiment with these to find the optimal capacity for your specific ODE problem. More complex ODEs might require deeper or wider networks.
Dependencies
Python 3.x
PyTorch (torch)
NumPy (numpy)
SciPy (scipy)
Matplotlib (matplotlib)
Future Work and Enhancements
Full Physics-Informed Neural Networks (PINNs): Extend the loss function to include the ODE residual throughout the domain, potentially improving generalization and extrapolation capabilities. (DONE)
Higher-Order ODEs: Generalize the ode_system function and solution handling for ODEs of arbitrary order.
Parameter Inference: Implement capabilities to infer ODE parameters from noisy data.
More Complex Boundary Conditions: Support for boundary value problems (BVPs) and more complex initial/boundary conditions.
Hyperparameter Optimization: Integrate tools for automated hyperparameter tuning (e.g., Optuna, Weights & Biases).
User Interface: Develop a more user-friendly graphical interface.