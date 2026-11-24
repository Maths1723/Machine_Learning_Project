# PINN follow-up (2026)

This folder now also contains a scriptable PINN library and a benchmark, written
after the Exercise 7 report to understand why the original PINN failed.
Short version (details and numbers in `notes/research_log.md`):

* The original PINN imposed the ODE only on t in [0, 1] (bug, fixed in
  `main.py`), ignored the alpha/beta weights, and was compared against a SciPy
  reference that was wrong for t < c. With those fixed, the PINN still gets
  stuck on Van der Pol: u = 0 is an exact solution, and with soft initial
  conditions it is a deep local minimum (*trivial-solution trap*).
* All configs are autonomous, so y(t; c) = Y(t - c): the whole c-family is one
  shifted trajectory. Building this into the network, u(t, c) = N(t - c), together
  with exactly enforced initial conditions, solves every c-family on every seed
  (relative L2 error 1e-4 to 1e-3 on pendulum and Rayleigh, <= 1.4e-2 on Van der Pol).

Summer 2026 update: the u = 0 trap turned out to be known in the literature
(see `notes/related_work.md`); on families *without* the symmetry, exact ICs
with a bounded profile make the PINN competitive with the report's data-driven
surrogate (better on VdP with varying mu, on par elsewhere) at ~20x the compute.
Stiff Van der Pol (mu up to 5) is still unsolved. Full tables: `notes/research_log.md`.

| file | what |
|---|---|
| `main.py` | interactive tool; `(p)` -> `(i)mproved` uses the new PINN, `(t)` plots the true error |
| `pinn_core.py` | models (`soft`, `hard`, `shift`, `shift-hard`), residual, causal/curriculum training, L-BFGS, evaluation |
| `numerics.py` | DOP853 reference solution (forward and backward from the IC point) |
| `parametric_*.py` | families where the parameter enters the ODE (VdP mu, pendulum damping) |
| `experiments/run_benchmark.py` | reproducible benchmark -> `results/benchmark.csv` |
| `experiments/summarize.py` | table + figure from the CSV |
| `tests/` | `pytest tests` |
| `surrogate.py` | the report's data-driven model as a baseline (`hybrid-N` in the benchmark) |
| `forced_pendulum.py`, `parametric_vdp_mu_stiff.py` | summer 2026 configs: broken symmetry, stiff regime |
| `experiments/make_figures.py` | `results/figures/` |
| `notes/` | `research_log.md` (dated results), `related_work.md`, `paper_outline.md` |

```bash
pip install -r requirements.txt
pytest tests
python experiments/run_benchmark.py --configs data_augmentation_van_der_pol.py     --models soft,shift-hard --epochs 6000 --lbfgs 1000 --curriculum 0.6 --seeds 0,1,2
python experiments/summarize.py results/benchmark.csv --plot results/error_vs_param.png
```

---

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

```
.
├── main.py                     # Main script for data generation, training, and evaluation.
├── SIMPLE_ode_support.py       # Core utilities: NN model, CustomLoss, Dataset, ODE solver wrapper.
└── *.py                        # ODE configs (one per problem).
```


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

```bash
python main.py
```

1. **Choose an ODE config** by number from the list, or type a path to another config file (`q` quits).
2. **Choose an action:**

| key | action |
|---|---|
| `p` | PINN training without data: `(o)riginal` or `(i)mproved` |
| `g` | generate or load a `.mat` dataset, then train or load the data-driven model |
| `n` | add noise to a clean dataset |
| `d` | train a network on noisy data |
| `o` | choose another ODE config |
| `q` | quit |

**PINN (`p`).** The original method asks for the loss weights, epochs, learning rate
and number of parameter values. The improved method asks for Adam epochs (default
6000) and L-BFGS iterations (default 1000). Afterwards you can plot a single solution
(`s`), the flux (`f`), the loss-based error norm (`e`) or the true error against SciPy (`t`).

**Data-driven (`g`).** Trains for 200 epochs, printing the train loss every epoch and
the test loss every 10. Afterwards you can plot a single solution (`s`), the flux (`f`)
or the error norm (`e`); predictions are compared with a fresh SciPy solution.
In `main.py` the loss for this model is plain MSE: the local `CustomODELoss` ignores
the boundary terms described above.

Parameter values outside the training range are extrapolation and can be far off.

### Adding a new ODE

Add a `.py` file to this folder (it then appears in the menu) defining:

* `PARAM_NAME_IN_MAT_FILE` (use `'c_values'` when the parameter is the initial-condition point),
  `PARAM_MIN_VALUE`, `PARAM_MAX_VALUE`, `NUM_PARAM_VALUES_GEN`
* `FIXED_Y0`, `FIXED_Y_PRIME0`
* `X_EVAL_START_GEN`, `X_EVAL_END_GEN`, `NUM_X_EVAL_POINTS_GEN`
* `X_START_IC_POINT`, only if the initial conditions are at a fixed point
* `ode_system(x, y, param_val)`, returning `[dy/dx, d2y/dx2]` with `y = [y, y']`

## Future Work and Enhancements

* Full Physics-Informed Neural Networks (PINNs): Extend the loss function to include the ODE residual throughout the domain, potentially improving generalization and extrapolation capabilities. (DONE)
* Higher-Order ODEs: Generalize the ode_system function and solution handling for ODEs of arbitrary order.
* Parameter Inference: Implement capabilities to infer ODE parameters from noisy data.
* More Complex Boundary Conditions: Support for boundary value problems (BVPs) and more complex initial/boundary conditions.
* Hyperparameter Optimization: Integrate tools for automated hyperparameter tuning (e.g., Optuna, Weights & Biases).
* User Interface: Develop a more user-friendly graphical interface.
