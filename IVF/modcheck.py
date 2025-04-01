import torch
from GATModel import *

# Load model architecture
model = GNNModel(indim=12, outdim=32, edge_dim=4, heads=6, dropout=0.4)  # Use the same hyperparameters!

# Load trained weights
model.load_state_dict(torch.load("model_files/model_dca_moreevt_2103_best.pth", map_location=torch.device('cpu')))

# Move model to evaluation mode
model.eval()

# Inspect learnable parameters
print(f"skip_weight: {model.skip_weight.item()}")
print(f"alpha: {model.alpha.item()}")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

