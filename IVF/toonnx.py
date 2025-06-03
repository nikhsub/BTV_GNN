import warnings
warnings.filterwarnings("ignore")

import torch
from GCNModel import *
import argparse
import onnxruntime as ort
import numpy as np

parser = argparse.ArgumentParser("GNN testing")

parser.add_argument("-lm",  "--load_model", default="", help="Load model file")
parser.add_argument("-o", "--output", default="testmod", help="Name of output onnx file")
args = parser.parse_args()


trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']
edge_features = ['dca', 'deltaR', 'dca_sig', 'cptopv', 'pvtoPCA_1', 'pvtoPCA_2', 'dotprod_1', 'dotprod_2', 'pair_mom', 'pair_invmass']

model = GNNModel(indim=len(trk_features), outdim=48, edge_dim=len(edge_features))
model.load_state_dict(torch.load(args.load_model, map_location=torch.device('cpu')))
model.eval()

batch_size = 1
num_nodes = 100  # Example number of nodes
num_edges = 300  # Example number of edges
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Random example inputs
x_in = torch.randn((batch_size, num_nodes, len(trk_features)), dtype=torch.float32, device=device)
edge_index = torch.randint(0, num_nodes, (batch_size, 2, num_edges), dtype=torch.float32, device=device)
edge_attr = torch.randn((batch_size, num_edges, len(edge_features)), dtype=torch.float32, device=device)

model.to(device)

print(f"x_in shape: {x_in.shape}")
print(f"edge_index shape: {edge_index.shape}")
print(f"edge_attr shape: {edge_attr.shape}")

# Export to ONNX
outname = args.output + ".onnx"
with torch.no_grad():
    torch.onnx.export(
        model,
        (x_in, edge_index, edge_attr),
        outname,
        opset_version=15,
        input_names=["x_in", "edge_index", "edge_attr"],
        output_names=["xf", "node_probs"],
        dynamic_axes={
            "x_in": {0: "batch_size", 1: "num_nodes"},
            "edge_index": {0: "batch_size", 2: "num_edges"},
            "edge_attr": {0: "batch_size", 1: "num_edges"}
        }
    )

print(f"ONNX model saved to {outname}")

session = ort.InferenceSession(outname)

# Prepare input dictionary for ONNX (must use .cpu().numpy())
onnx_inputs = {
    "x_in": x_in.cpu().numpy(),
    "edge_index": edge_index.cpu().numpy(),  # ensure int64
    "edge_attr": edge_attr.cpu().numpy(),
}

# Run inference with ONNX model
onnx_outputs = session.run(None, onnx_inputs)
onnx_node_probs = torch.tensor(onnx_outputs[1])

# Get outputs from PyTorch model again
with torch.no_grad():
    _, torch_node_probs = model(x_in, edge_index, edge_attr)  # force edge_index to long

# Compare outputs
abs_diff = torch.abs(torch_node_probs.cpu() - onnx_node_probs.cpu())
max_diff = abs_diff.max().item()
mean_diff = abs_diff.mean().item()

print(f"ONNX vs PyTorch consistency check:")
print(f"Max difference:  {max_diff:.6e}")
print(f"Mean difference: {mean_diff:.6e}")


