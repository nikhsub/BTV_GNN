import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pickle5 as pickle
import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader
import argparse
from GATModel import *
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap
class GNNWrapper(torch.nn.Module):
    def __init__(self, model, edge_index, edge_attr):
        super(GNNWrapper, self).__init__()
        self.model = model
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def forward(self, x):
        """
        Takes only `x` (node features) as input, while using fixed `edge_index` and `edge_attr`.
        """
        num_nodes = x.shape[0]

        # Ensure edge_index is within bounds
        valid_edges = (self.edge_index[0] < num_nodes) & (self.edge_index[1] < num_nodes)
        edge_index = self.edge_index[:, valid_edges]
        edge_attr = self.edge_attr[valid_edges]

        _, logits, _, _, _ = self.model(x, edge_index, edge_attr)
        return torch.sigmoid(logits)  # Apply sigmoid because it's a single-output model

parser = argparse.ArgumentParser("GNN testing")

parser.add_argument("-f", "--testfile", default="", help="Load testing data from a file")
parser.add_argument("-m",  "--model", default="", help="Load model file")

args = parser.parse_args()

model = GNNModel(indim=12, outdim=32, edge_dim=4, heads=6, dropout=0.4)

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']

# Load trained weights
model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
model.eval()  # Set to evaluation mode

if args.testfile != "":
    print(f"Loading testing data from {args.testfile}...")
    with open(args.testfile, 'rb') as f:
        train_graphs = pickle.load(f)

val_graphs = train_graphs[:10]

for evt, data in enumerate(val_graphs):


    wrapped_model = GNNWrapper(model, data.edge_index, data.edge_attr) 
        
    explainer = shap.GradientExplainer(wrapped_model, data.x)

    shap_values = explainer.shap_values(data.x)

# Convert SHAP values to NumPy array
    shap_values = np.array(shap_values)
    
    # Ensure shape is (310, 12) for feature importance
    shap_values = shap_values.squeeze(-1)
    
    print("Fixed SHAP Values shape:", shap_values.shape)  # Should now be (310, 12)


    print(f"Generating SHAP plots for Event {evt}...")

    # Save Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, data.x.numpy(), feature_names=trk_features, show=False)
    plt.savefig(f"shap_summary_event_{evt}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save Dependence Plot
    plt.figure()
    shap.dependence_plot(0, shap_values, data.x.numpy(), feature_names=trk_features, show=False)
    plt.savefig(f"shap_dependence_event_{evt}.png", dpi=300, bbox_inches="tight")
    plt.close()    
        
