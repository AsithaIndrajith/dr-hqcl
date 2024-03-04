# A Fusion of Supervised Contrastive Learning and Variational Quantum Classifiers

In medical applications, machine learning often grapples with limited training data. Classical self-supervised deep learning techniques have been helpful in this domain, but these algorithms have yet to achieve the required accuracy for medical use. Recently quantum algorithms show promise in handling complex patterns with small datasets. To address this challenge, this study presents a novel solution that combines self-supervised learning with Variational Quantum Classifiers (VQC) and utilizes Principal Component Analysis (PCA) as the dimensionality reduction technique. This unique approach ensures generalization even with a small training dataset while preserving data privacy, a vital consideration in medical applications. PCA is effectively utilized for dimensionality reduction, enabling VQC to operate with just 2 Q-bits, overcoming current quantum hardware limitations, and gaining an advantage over classical methods. The proposed model was benchmarked against linear classification models using diverse public image datasets to validate its effectiveness. The results demonstrate remarkable accuracy, with achievements of 90% on PneumoniaMNIST, 90% on BreastMNIST, 80% on PathMNIST, and 80% on ChestMNIST medical datasets. Additionally, for non-medical datasets, the model attained 85% on Hymenoptera Ants & Bees and 90% on the Kaggle Cats & Dogs dataset.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Published Research Paper](#published-research-paper)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

The motivations driving this research, stem from various factors including the limitations of medical data availability, the shortcomings of existing self-supervised and quantum self-supervised learning approaches in terms of generalizability with limited data, and the inadequate accuracy achieved by quantum self-supervised learning methods even when applied to large datasets.

## Features

- **Efficient medical image classification with limited labeled data:**
  A novel generalized model is proposed for classifying medical image data with a minimal number of labeled images. The model's efficiency is empirically validated, demonstrating its potential to handle data scarcity.

- **Enhancing model accuracy with data pre-processing techniques:**
  A series of data pre-processing techniques are developed to improve the model's accuracy.

- **Enhancing quantum model efficiency:**
  To overcome the limitations posed by constrained quantum hardware resources, a combination of principal component analysis and variational quantum classifier is introduced. This integration enhances the practicality and flexibility of the proposed model.

## Getting Started

To get started with this project, follow the steps below to set up and run the code on your local machine.

### Prerequisites

Before running the code, make sure you have the following software and dependencies installed:

1. **Conda:**
   - If you don't have Conda installed, follow the installation instructions for your operating system [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. **Create a Conda Environment:**
   - Create a new Conda environment using the provided `environment.yml` file:
     ```bash
     conda env create -f environment.yml
     ```

   - Activate the Conda environment:
     ```bash
     conda activate your_environment_name
     ```

3. **PyTorch:**
   - Install PyTorch within your Conda environment:
     ```bash
     conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch
     ```

4. **IBM Qiskit:**
   - Install IBM Qiskit within your Conda environment:
     ```bash
     conda install -c conda-forge qiskit
     ```

   - Additional installation for IBM Qiskit Aer (optional):
     ```bash
     conda install -c conda-forge qiskit-aer
     ```

   - Additional installation for IBM Qiskit Aqua (optional):
     ```bash
     conda install -c conda-forge qiskit-aqua
     ```

Make sure to replace `your_environment_name` with the desired name for your Conda environment.

Now, you're ready to run the project with the specified dependencies.

### Installation

To install and run the project on your local machine, follow these simple steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/AsithaIndrajith/dr-hqcl.git

## Usage

To use this project, follow the steps outlined below:

1. **Open the Jupyter Notebook:**
   - Navigate to the project directory using the terminal or file explorer.
   - Launch the Jupyter Notebook by running:
     ```bash
     jupyter notebook
     ```
   - In the Jupyter interface, open the `main.ipynb` notebook.

2. **Run the Code:**
   - Within the Jupyter Notebook, navigate to the specific code cell you want to run.
   - Execute the cell by clicking the "Run" button or using the keyboard shortcut `Shift + Enter`.

Note: Ensure that you have all the necessary dependencies installed in your Python environment. If you encounter any issues, refer to the project documentation or README for additional instructions.

## Contributing

We welcome contributions from the community to enhance and improve this project. If you'd like to contribute, please follow these guidelines:

### Reporting Issues

If you encounter any bugs, issues, or have suggestions for improvements, please check if the issue has already been reported in the [Issues](https://github.com/AsithaIndrajith/dr-hqcl/issues) section. If not, feel free to open a new issue with details on the problem or enhancement request.

### Feature Requests

If you have ideas for new features or improvements, you can submit them by opening an issue in the [Issues](https://github.com/AsithaIndrajith/dr-hqcl/issues) section. Provide a clear description of the proposed feature and its benefits.

### Pull Requests

We encourage you to contribute directly by submitting pull requests. Follow these steps:

1. Fork the repository to your GitHub account.
2. Clone your forked repository to your local machine:
   ```bash
   git clone https://github.com/AsithaIndrajith/dr-hqcl.git
   ```
3. Create a new branch for your changes:
   ```bash
   git checkout -b feature_branch_name
   ```
4. Make your changes, commit them, and push to your forked repository:
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin feature_branch_name
   ```
5. Open a pull request on the [Pull Requests](https://github.com/AsithaIndrajith/dr-hqcl/pulls) page. Provide a clear title and description for your changes.

## Published Research Paper

The code in this repository is associated with the following published research paper:

- Author(s): Asitha Kottahachchi Kankanamge Don; Ibrahim Khalil; Mohammed Atiquzzaman
- Title: A Fusion of Supervised Contrastive Learning and Variational Quantum Classifiers
- Journal/Conference: IEEE TRANSACTIONS ON CONSUMER ELECTRONICS
- Year: 2024
- DOI or Link: https://doi.org/10.1109/TCE.2024.3351649

## License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work is supported by the Australian Research Council Discovery Project (DP210102761). We would like to extend our acknowledgment to Robert Shen from RACE (RMIT AWS Cloud Supercomputing Hub) and Jeff Paine from Amazon Web Services (AWS) for their invaluable provision of computing resources.
