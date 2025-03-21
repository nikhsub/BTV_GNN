import torch
from GATModel import *

# Load model architecture
model = GNNModel(indim=12, outdim=32, edge_dim=4, heads=8, dropout=0.25)  # Use the same hyperparameters!

# Load trained weights
model.load_state_dict(torch.load("model_files/model_dca_1703_best.pth"))

# Move model to evaluation mode
model.eval()

# Inspect learnable parameters
print(f"skip_weight: {model.skip_weight.item()}")
print(f"alpha: {model.alpha.item()}")
