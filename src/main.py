# cli.py

import argparse
import os
import torch
from torch.utils.data import DataLoader

from models import Encoder, Generator, GNNModel
from dataset import CustomDataset
from utils import collate_fn
from train import train_model
import config


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # For illustration, create synthetic data
    n_samples = config.NUM_SAMPLES
    p = config.INPUT_DIM
    d_list = config.MODALITY_DIMS
    k = config.LATENT_DIM

    torch.manual_seed(0)  # For reproducibility

    X = torch.randn(n_samples, p)
    N_list = [torch.randn(n_samples, d) for d in d_list]
    M_list = [torch.randint(0, 2, (n_samples, d)) for d in d_list]
    Y = torch.randn(n_samples)

    # Create Dataset and DataLoader
    dataset = CustomDataset(X, N_list, M_list, Y)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Initialize Models
    encoder = Encoder(input_dim=p, latent_dim=k, hidden_dim=config.HIDDEN_DIM)
    generators = [Generator(latent_dim=k, output_dim=d, hidden_dim=config.HIDDEN_DIM) for d in d_list]
    gnn_model = GNNModel(input_dim=sum(d_list), hidden_dim=config.HIDDEN_DIM)

    # Train the Model
    train_model(
        encoder, generators, gnn_model, dataloader, device,
        num_epochs=config.NUM_EPOCHS,
        lambda_rec=config.LAMBDA_REC,
        learning_rate=config.LEARNING_RATE
    )

if __name__ == "__main__":
    main()


# def main():
#     parser = argparse.ArgumentParser(description=f'GenMM-GNN: Generative Imputation of Missing Modalities using Neural Networks')
    
#     # Data Arguments
#     parser.add_argument('-output_dir', type=str, default='./output',
#                         help="Output directory (default: './output').")
#     parser.add_argument('-input', type=str, required=True,
#                         help="Input data file in csv, txt, or npz format (required).")
#     parser.add_argument('-dataset', type=str, default='MyData',
#                         help="Dataset name (default: 'MyData').")
#     parser.add_argument('--save_model', default=False, action='store_true',
#                         help="Whether to save the trained model (default: False).")
#     parser.add_argument('--save_res', default=False, action='store_true',
#                         help="Whether to save results during training (default: False).")
    
#     # Model Hyperparameters
#     parser.add_argument('-input_dim', type=int, default=50,
#                         help="Dimension of input covariates X (default: 50).")
#     parser.add_argument('-latent_dim', type=int, default=16,
#                         help="Dimension of latent space Z (default: 16).")
#     parser.add_argument('-hidden_dim', type=int, default=128,
#                         help="Number of units in hidden layers (default: 128).")
#     parser.add_argument('-modality_dims', type=int, nargs='+', default=[30, 40, 20],
#                         help="Dimensions of modalities N1, N2, N3, ... (default: [30, 40, 20]).")
#     parser.add_argument('--use_z_rec', default=True, action='store_true',
#                         help="Use reconstruction for latent features (default: True).")
    
#     # Training Hyperparameters
#     parser.add_argument('-batch_size', type=int, default=64,
#                         help="Batch size (default: 64).")
#     parser.add_argument('-num_epochs', type=int, default=10,
#                         help="Number of epochs (default: 10).")
#     parser.add_argument('-learning_rate', type=float, default=0.001,
#                         help="Learning rate (default: 0.001).")
#     parser.add_argument('-lambda_rec', type=float, default=1.0,
#                         help="Coefficient for reconstruction loss (default: 1.0).")
#     parser.add_argument('-seed', type=int, default=123,
#                         help="Random seed for reproducibility (default: 123).")
    
#     # Parse arguments
#     args = parser.parse_args()
#     params = vars(args)
    
#     # Update config with command-line arguments
#     config.INPUT_DIM = args.input_dim
#     config.LATENT_DIM = args.latent_dim
#     config.HIDDEN_DIM = args.hidden_dim
#     config.MODALITY_DIMS = args.modality_dims
#     config.BATCH_SIZE = args.batch_size
#     config.NUM_EPOCHS = args.num_epochs
#     config.LEARNING_RATE = args.learning_rate
#     config.LAMBDA_REC = args.lambda_rec
#     config.SEED = args.seed
#     config.DATASET_NAME = args.dataset
#     config.OUTPUT_DIR = args.output_dir
#     config.SAVE_MODEL = args.save_model
#     config.SAVE_RES = args.save_res
#     config.USE_Z_REC = args.use_z_rec
    
#     # Set random seed
#     torch.manual_seed(config.SEED)
    
#     # Set device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Load data from file
#     X, N_list, M_list, Y = load_data(args.input)
    
#     # Create Dataset and DataLoader
#     dataset = CustomDataset(X, N_list, M_list, Y)
#     dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
#     # Initialize Models
#     encoder = Encoder(input_dim=config.INPUT_DIM, latent_dim=config.LATENT_DIM, hidden_dim=config.HIDDEN_DIM)
#     generators = [Generator(latent_dim=config.LATENT_DIM, output_dim=d, hidden_dim=config.HIDDEN_DIM) for d in config.MODALITY_DIMS]
#     gnn_model = GNNModel(input_dim=sum(config.MODALITY_DIMS), hidden_dim=config.HIDDEN_DIM)
    
#     # Train the Model
#     print('Start training...')
#     train_model(
#         encoder, generators, gnn_model, dataloader, device,
#         num_epochs=config.NUM_EPOCHS,
#         lambda_rec=config.LAMBDA_REC,
#         learning_rate=config.LEARNING_RATE,
#         save_results=config.SAVE_RES,
#         use_z_rec=config.USE_Z_REC
#     )
    
#     # Save the model if required
#     if config.SAVE_MODEL:
#         # Create the output directory if it doesn't exist
#         os.makedirs(config.OUTPUT_DIR, exist_ok=True)
#         # Save the models
#         torch.save(encoder.state_dict(), os.path.join(config.OUTPUT_DIR, 'encoder.pth'))
#         for i, gen in enumerate(generators):
#             torch.save(gen.state_dict(), os.path.join(config.OUTPUT_DIR, f'generator_{i}.pth'))
#         torch.save(gnn_model.state_dict(), os.path.join(config.OUTPUT_DIR, 'gnn_model.pth'))
#         print(f'Models saved to {config.OUTPUT_DIR}')

# def load_data(file_path):
#     # Implement your data loading logic here
#     # For example, if the data is stored in a CSV file
#     import pandas as pd
#     import numpy as np
    
#     # Placeholder implementation
#     # Replace with actual data loading and preprocessing
#     if file_path.endswith('.csv') or file_path.endswith('.txt'):
#         data = pd.read_csv(file_path)
#         # Process data to extract X, N_list, M_list, Y
#         # ...
#     elif file_path.endswith('.npz'):
#         data = np.load(file_path)
#         # Process data to extract X, N_list, M_list, Y
#         # ...
#     else:
#         raise ValueError("Unsupported file format. Please provide a .csv, .txt, or .npz file.")
    
#     # For demonstration purposes, create random tensors
#     n_samples = len(data)
#     X = torch.randn(n_samples, config.INPUT_DIM)
#     N_list = [torch.randn(n_samples, d) for d in config.MODALITY_DIMS]
#     M_list = [torch.randint(0, 2, (n_samples, d)) for d in config.MODALITY_DIMS]
#     Y = torch.randn(n_samples)
    
#     return X, N_list, M_list, Y

# if __name__ == '__main__':
#     main()