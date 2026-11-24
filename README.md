# Numerical Analysis Laboratory - Exercises 7 & 9: Neural Networks in Differential Problems and Multiclass Image Classification

This repository contains the materials for "Numerical Analysis Laboratory - Exercise 7 & 9," a combined project that explores the application of Neural Networks to two distinct scientific computing challenges: solving differential equations and performing image classification. This work was completed in collaboration with Chiara Galimberti.

## Project Overview

This paper is structured around two main exercises, each addressing a significant application of neural networks:

### Exercise 1: Neural Networks in Differential Problems
**Description:** This work thoroughly investigates the use of neural networks (NNs) for solving non-linear second-order ordinary differential equations (ODEs). We first address the challenges associated with Physics-Informed Neural Networks (PINNs), noting their tendency to get "stuck" on qualitatively incorrect solutions despite low error functionals, particularly for oscillatory problems. We then propose and evaluate a comparison with hybrid numerical-aided Neural Network models. These hybrid models, trained on numerically generated solutions, proved to be more robust, even if slightly less precise over short intervals.

The study further explores the potential applications of these hybrid models in generalization, data augmentation, denoising, and data compression. We demonstrate that even minimal data augmentation can drastically improve generalization beyond training data. The approach involved generating data via traditional numerical solvers (e.g., SciPy's `solve_ivp`), training a fully-connected NN to map parameters to ODE solutions, and then inferring solutions for unseen parameters. We applied this methodology to classic non-linear ODEs: the Damped Pendulum, Van der Pol, and Rayleigh equations. Preliminary findings also showcase the NNs' capability for data compression (up to 10x reduction) and inherent denoising effects.

**Key Competencies:** Neural Networks (PINNs, Hybrid Models), Ordinary Differential Equations (ODEs), Numerical Analysis, Generalization, Data Augmentation, Data Compression, Data Denoising, Python Programming.

### Exercise 2: Multiclass Image Classification of Fruits-360 Dataset
**Description:** This exercise tackles the problem of multiclass image classification using neural networks, specifically on the diverse Fruits-360 dataset. We implemented and rigorously evaluated two fundamental neural network architectures: a Multilayer Perceptron (MLP) and a Convolutional Neural Network (CNN).

Performance was assessed by varying key hyperparameters, primarily the number of layers and training epochs, across two distinct experimental settings:
1.  **Classifying Different Types of Fruits:** Focusing on distinguishing broadly different fruit categories (e.g., Apple vs. Banana).
2.  **Classifying Different Varieties of the Same Fruit:** Challenging the networks to differentiate subtle variations within a single fruit type (e.g., Apple Braeburn vs. Apple Golden).

Our results highlight the varying degrees of accuracy achieved by both architectures under different configurations, showcasing their respective strengths and weaknesses for general and fine-grained image classification tasks. The CNN generally outperformed the MLP on the full dataset, demonstrating its superiority in learning spatial hierarchies from image data.

**Key Competencies:** Machine Learning, Deep Learning, Image Classification, Multilayer Perceptrons (MLP), Convolutional Neural Networks (CNN), Hyperparameter Tuning, Dataset Analysis, Python Programming.

## Files in this Repository

* `Fusar_Galimberti_Prog_Lan.pdf`: The complete technical report.
* `PINN_and_ODEs/ODEs/`: Hybrid (data-driven) neural networks for ODEs (`simple_ode.py` plus one config file per problem).
* `PINN_and_ODEs/PINN/`: PINNs, the 2026 follow-up library, benchmarks, results and notes.
* `Fruits/`: Multiclass image classification (MLP and CNN).

## How to Run

### Neural networks for ODEs

```bash
pip install -r PINN_and_ODEs/PINN/requirements.txt
cd PINN_and_ODEs/PINN && python main.py        # PINN (interactive)
cd PINN_and_ODEs/ODEs && python simple_ode.py  # hybrid models (interactive)
```

Run each `cd` line from the repository root.

### Image classification

Requires `torch`, `torchvision`, `numpy` and `matplotlib`, and a local copy of the
Fruits-360 dataset; set the dataset paths in `Fruits/FRUTTA_main.py`, then:

```bash
cd Fruits && python FRUTTA_main.py
```

## Technologies Utilized

* **Python** with PyTorch and Matplotlib.
* **SciPy**: numerical solvers for data generation and reference solutions.

## 2026 follow-up: PINNs

The PINN part of Exercise 1 was revisited in spring 2026: several bugs in the
original PINN were fixed (residual only enforced on [0, 1], ignored loss
weights, wrong SciPy reference for t < c), and a new PINN with exact initial
conditions and built-in time-translation symmetry was benchmarked. It solves
all three oscillator families reliably, including Van der Pol where the
original approach got stuck on the trivial solution. See
`PINN_and_ODEs/PINN/README.md` and `PINN_and_ODEs/PINN/notes/research_log.md`.

## Authors

* **Leonardo Fusar Bassini**
* **Chiara Galimberti**

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
