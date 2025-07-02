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

## Files in this Subdirectory

* `Laboratorio_Analisi_Numerica_Esercizio_7_9.pdf`: The complete technical report in PDF format.
* `Laboratorio_Analisi_Numerica_Esercizio_7_9.tex`: The LaTeX source code for this report.
* `img/`: Directory containing all figures and images used in the report.
* `src/ode_nn/`: Python scripts for Neural Networks in Differential Problems (e.g., `main_ode.py`, model definitions).
* `src/image_classification/`: Python scripts for Multiclass Image Classification (e.g., `mlp_classifier.py`, `cnn_classifier.py`).
* `data/`: (If applicable) Directory for any datasets used (e.g., a subset of Fruits-360 if not downloaded on-the-fly).
* `sample.bib`: The BibTeX file containing references cited in the report.

## How to Run

### For Neural Networks in Differential Problems (Python):
1.  Ensure you have Python installed (preferably Python 3.x).
2.  Navigate to `src/ode_nn/` within this sub-repository.
3.  Install necessary Python dependencies (e.g., `numpy`, `scipy`, `tensorflow` or `pytorch` - depending on which framework you used for NNs). It's recommended to include a `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the main Python script (e.g., `python main_ode.py`) from your terminal.

### For Multiclass Image Classification (Python):
1.  Ensure you have Python installed.
2.  Navigate to `src/image_classification/` within this sub-repository.
3.  Install necessary Python dependencies (e.g., `torch` or `tensorflow`, `torchvision` or `keras`, `matplotlib`, `scikit-learn`). A `requirements.txt` file is highly recommended.
    ```bash
    pip install -r requirements.txt
    ```
4.  Ensure the Fruits-360 dataset is accessible (it might be downloaded automatically by your script, or you may need to place it in the `data/` directory).
5.  Run the respective Python scripts (e.g., `python mlp_classifier.py` or `python cnn_classifier.py`) from your terminal.

## Technologies Utilized

* **Python:** For Neural Network implementations and data processing (TensorFlow/Keras or PyTorch).
* **MATLAB:** (If any legacy or comparative code from other projects was leveraged, otherwise can be omitted if not directly used here).
* **SciPy:** For numerical solvers used in data generation for ODEs.
* **C/C++:** (Majorly for testing purposes within the broader course context, though not directly applied in these specific Python scripts).

## Authors

* **Leonardo Fusar Bassini** - [Link to Leonardo's LinkedIn profile]
* **Chiara Galimberti** - [Link to Chiara's profile, if available]

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
