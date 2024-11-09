# config.py

# Model Hyperparameters
INPUT_DIM = 50       # Dimension of X
LATENT_DIM = 16      # Dimension of latent space Z
HIDDEN_DIM = 128     # Hidden layer size for models
MODALITY_DIMS = [30, 40, 20]  # Dimensions of modalities N1, N2, N3, ...

# Training Hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
LAMBDA_REC = 1.0     # Weight for reconstruction loss

# Data Parameters
NUM_SAMPLES = 10000  # Number of samples in synthetic data