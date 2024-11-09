# GenMM-GNN

**Generative Modeling for Missing Modalities using Graph Neural Networks**

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Using Conda Environment](#using-conda-environment)
- [Usage](#usage)
  - [Command-Line Interface](#command-line-interface)
  - [Example Command](#example-command)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

**GenMM-GNN** (Generative Modeling for Missing Modalities using Graph Neural Networks) is a comprehensive framework designed to handle datasets with multiple modalities, some of which may have missing blocks of data. By leveraging generative models, GenMM-GNN encodes high-dimensional covariates into a lower-dimensional latent space and generates the missing modalities. These modalities, both real and generated, are then integrated using a Graph Neural Network (GNN) to predict outcomes effectively.

This framework is particularly useful in scenarios where data is incomplete or partially missing across different modalities, enabling robust predictive performance by imputing and leveraging all available information.

## Features

- **Generative Modeling**: Encodes covariates into a latent space and generates missing modalities.
- **Graph Neural Networks**: Integrates real and generated modalities using GNNs for outcome prediction.
- **Modular Codebase**: Organized into separate modules for easy maintenance and scalability.
- **Command-Line Interface**: Flexible configuration through command-line arguments.
- **Conda Environment Setup**: Simplifies dependency management with a ready-to-use `environment.yml`.

## Project Structure

The project is organized into the following files and directories to ensure clarity and maintainability:

## Installation

### Prerequisites

- **Operating System**: Unix-based system (e.g., Linux).
- **Python Version**: Python 3.6.
- **Package Manager**: Conda.



### Using Conda Environment

To ensure all dependencies are correctly installed and compatible, it's recommended to use the provided Conda environment configuration.

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/GenMM-GNN.git
   cd GenMM-GNN

   conda env create -f environment.yml
   conda activate GenMM-GNN-env
   ```

2. **Usage**
GenMM-GNN can be configured and run using command-line arguments. Below are instructions on how to use the command-line interface provided by cli.py.

Command-Line Interface
The project provides a cli.py script that parses command-line arguments to configure and initiate the training process.

Available Arguments
Data Arguments:

-output_dir: Output directory (default: ./output).
-input: Input data file in csv, txt, or npz format (required).
-dataset: Dataset name (default: MyData).
--save_model: Flag to save the trained model (default: False).
--save_res: Flag to save results during training (default: False).
Model Hyperparameters:

-input_dim: Dimension of input covariates X (default: 50).
-latent_dim: Dimension of latent space Z (default: 16).
-hidden_dim: Number of units in hidden layers (default: 128).
-modality_dims: Dimensions of modalities N1, N2, N3, ... (default: [30, 40, 20]).
--use_z_rec: Use reconstruction loss for latent features (default: True).
Training Hyperparameters:

-batch_size: Batch size for training (default: 64).
-num_epochs: Number of training epochs (default: 10).
-learning_rate: Learning rate for the optimizer (default: 0.001).
-lambda_rec: Coefficient for reconstruction loss (default: 1.0).
-seed: Random seed for reproducibility (default: 123).
Example Command
bash
Copy code
python cli.py -input data.csv -output_dir ./results -num_epochs 20 --save_model --save_res
Explanation:

-input data.csv: Specifies the input data file.
-output_dir ./results: Sets the output directory to ./results.
-num_epochs 20: Trains the model for 20 epochs.
--save_model: Saves the trained models after training.
--save_res: Saves intermediate results during training.
Example Usage
This section provides examples of how to use the GenMM-GNN project. Follow the instructions below to run the training script with various configurations.

Prerequisites
Conda Environment: Ensure that you have created and activated the Conda environment as specified in the Installation section.

bash
Copy code
conda activate GenMM-GNN-env
Data Preparation: Prepare your input data file in one of the supported formats (.csv, .txt, .npz). Ensure that the data is properly formatted according to the project's requirements.

Running the Training Script
Use the cli.py script to configure and initiate the training process. Below are example commands demonstrating different configurations.

1. Basic Training with Required Arguments
bash
Copy code
python cli.py -input data.csv -output_dir ./results
Arguments:

-input data.csv: Specifies the input data file in csv format.
-output_dir ./results: Sets the output directory to ./results.
Description:

This command runs the training process using the default settings for all optional parameters. The trained models and results will be saved in the ./results directory.

2. Training with Custom Number of Epochs and Saving Models
bash
Copy code
python cli.py -input data.csv -output_dir ./results -num_epochs 20 --save_model
Arguments:

-input data.csv: Specifies the input data file in csv format.
-output_dir ./results: Sets the output directory to ./results.
-num_epochs 20: Trains the model for 20 epochs.
--save_model: Saves the trained models after training.
Description:

This command configures the training to run for 20 epochs and saves the trained models in the specified output directory.

3. Training with Increased Batch Size and Reproducibility
bash
Copy code
python cli.py -input data.csv -output_dir ./output -batch_size 64 -num_epochs 30 -seed 42
Arguments:

-input data.csv: Specifies the input data file in csv format.
-output_dir ./output: Sets the output directory to ./output.
-batch_size 64: Sets the batch size to 64.
-num_epochs 30: Trains the model for 30 epochs.
-seed 42: Sets the random seed to 42 for reproducibility.
Description:

This command increases the batch size to 64, runs the training for 30 epochs, and sets a specific random seed to ensure reproducible results.

4. Training with Multiple Modalities and Advanced Settings
bash
Copy code
python cli.py -input dataset.npz -output_dir ./advanced_results -dataset "HealthData" -batch_size 128 -num_epochs 50 --save_model --save_res --use_z_rec --use_v_gan
Arguments:

-input dataset.npz: Specifies the input data file in npz format.
-output_dir ./advanced_results: Sets the output directory to ./advanced_results.
-dataset "HealthData": Names the dataset as "HealthData".
-batch_size 128: Sets the batch size to 128.
-num_epochs 50: Trains the model for 50 epochs.
--save_model: Saves the trained models after training.
--save_res: Saves intermediate results during training.
--use_z_rec: Enables reconstruction loss for latent features.
--use_v_gan: Enables GAN distribution matching for covariates.
Arguments:

-input dataset.npz: Specifies the input data file in npz format.
-output_dir ./advanced_results: Sets the output directory to ./advanced_results.
-dataset "HealthData": Names the dataset as "HealthData".
-batch_size 128: Sets the batch size to 128.
-num_epochs 50: Trains the model for 50 epochs.
--save_model: Saves the trained models after training.
--save_res: Saves intermediate results during training.
--use_z_rec: Enables reconstruction loss for latent features.
--use_v_gan: Enables GAN distribution matching for covariates.
Description:

This advanced command configures the training process to handle a dataset named "HealthData" with a larger batch size and more epochs. It also enables additional features such as reconstruction loss for latent features and GAN distribution matching for covariates, while saving both models and intermediate results.

Full Example Command
Combining multiple arguments to fully customize the training process:

bash
Copy code
python cli.py \
  -input data.csv \
  -output_dir ./results \
  -dataset "MyDataset" \
  -batch_size 64 \
  -num_epochs 25 \
  -learning_rate 0.001 \
  -lambda_rec 1.5 \
  -seed 123 \
  --save_model \
  --save_res \
  --use_z_rec \
  --use_v_gan
Arguments:

-input data.csv: Input data file.
-output_dir ./results: Output directory for results.
-dataset "MyDataset": Custom name for the dataset.
-batch_size 64: Batch size for training.
-num_epochs 25: Number of training epochs.
-learning_rate 0.001: Learning rate for the optimizer.
-lambda_rec 1.5: Coefficient for the reconstruction loss.
-seed 123: Random seed for reproducibility.
--save_model: Save the trained models.
--save_res: Save intermediate results.
--use_z_rec: Use reconstruction loss for latent features.
--use_v_gan: Use GAN distribution matching for covariates.
Description:

This comprehensive command sets various hyperparameters and options to tailor the training process precisely to your needs. It adjusts the learning rate, reconstruction loss coefficient, and enables advanced features, ensuring both the models and results are saved for future use.

Tips for Effective Usage
Ensure Data Compatibility: Before running the training script, verify that your input data is correctly formatted and compatible with the project's requirements.
Monitor Training Progress: Utilize the --save_res flag to save intermediate results, allowing you to monitor and evaluate the training progress.
Adjust Hyperparameters: Experiment with different hyperparameter settings (e.g., -batch_size, -num_epochs, -learning_rate) to optimize model performance for your specific dataset.
Reproducibility: Use the -seed argument to set a random seed, ensuring that your experiments are reproducible.
Resource Management: Adjust the -batch_size and -num_epochs based on your system's computational resources to balance training time and performance.