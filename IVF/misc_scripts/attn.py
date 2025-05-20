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

def plot_attention_weights(attn_weights, layer_name="Layer 1", event_num=0, save_dir="attention_plots"):
    """Plots and saves attention weight distribution for a given event."""
    os.makedirs(save_dir, exist_ok=True)  # Create directory if not exists

    attn_np = attn_weights.cpu().detach().numpy()  # Convert to NumPy
    num_heads = attn_np.shape[0]  # Get number of attention heads

    plt.figure(figsize=(10, 6))
    for head in range(num_heads):
        sns.histplot(attn_np[head].flatten(), bins=50, kde=True, label=f"Head {head+1}")

    plt.xlabel("Attention Weight Value")
    plt.ylabel("Frequency")
    plt.title(f"Attention Weights ({layer_name}, Event {event_num})")
    plt.legend()

    # Save the figure
    filename = f"{save_dir}/{layer_name.replace(' ', '_')}_event_{event_num}.png"
    plt.savefig(filename)
    plt.close()  # Prevents displaying in interactive environments
    print(f"Saved {filename}")

parser = argparse.ArgumentParser("GNN testing")

parser.add_argument("-f", "--testfile", default="", help="Load testing data from a file")
parser.add_argument("-m",  "--model", default="", help="Load model file")

args = parser.parse_args()

model = GNNModel(indim=12, outdim=32, edge_dim=4, heads=6, dropout=0.4)

# Load trained weights
model.load_state_dict(torch.load(args.model))
model.eval()  # Set to evaluation mode

if args.testfile != "":
    print(f"Loading testing data from {args.testfile}...")
    with open(args.testfile, 'rb') as f:
        train_graphs = pickle.load(f)

val_graphs = train_graphs[0:2]


for evt, data in enumerate(val_graphs):
    with torch.no_grad():  # Disable gradient computation for inference
        _, logits, attn1, attn2, attn3 = model(data.x, data.edge_index, data.edge_attr)
        preds = torch.sigmoid(logits)

        print("PREDS", preds)
        #print("Attention Weights from GAT Layer 1:", attn1)
        #print("Attention Weights from GAT Layer 2:", attn2)
        #print("Attention Weights from GAT Layer 3:", attn3)S
        plot_attention_weights(attn1[1], layer_name="Layer 1", event_num=evt)
        plot_attention_weights(attn2[1], layer_name="Layer 2", event_num=evt)
        plot_attention_weights(attn3[1], layer_name="Layer 3", event_num=evt)

        print("NEXT")
        print()

