import warnings
warnings.filterwarnings("ignore")

import torch
from GCNModel import *
import pickle
import argparse
import torch_geometric
from torch_geometric.data import Data, DataLoader
import onnxruntime as ort
import numpy as np
from torch.onnx import dynamo_export
from torch.onnx.verification import find_mismatch

parser = argparse.ArgumentParser("GNN testing")

parser.add_argument("-lm",  "--load_model", default="", help="Load model file")
parser.add_argument("-f",  "--file", default="", help="Load sample file")
parser.add_argument("-o", "--output", default="testmod", help="Name of output onnx file")
args = parser.parse_args()


trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']
edge_features = ['dca', 'deltaR', 'dca_sig', 'cptopv', 'pvtoPCA_1', 'pvtoPCA_2', 'dotprod_1', 'dotprod_2', 'pair_mom', 'pair_invmass']

model = GNNModel(indim=len(trk_features), outdim=48, edge_dim=len(edge_features))
model.load_state_dict(torch.load(args.load_model, map_location=torch.device('cpu')))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

print(f"Loading data from {args.file}...")
with open(args.file, 'rb') as f:
    graphs = pickle.load(f)

data = graphs[10]

# Random example inputs
x_in = data.x.unsqueeze(0).to(device)
edge_index = data.edge_index.unsqueeze(0).to(device)
edge_attr = data.edge_attr.unsqueeze(0).to(device)

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
        },
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )

#export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
#torch.onnx.OnnxRegistry.opset_version =15
#
#with torch.no_grad():
#    onnx_program = dynamo_export(model, 
#                                x_in, edge_index, edge_attr, 
#                                export_options=export_options)
#    onnx_program.save(args.output + ".onnx")
#
## Check for mismatches
#print("Running mismatch check...")
#mismatch = find_mismatch(model, (x_in, edge_index, edge_attr), onnx_program)
#
#if mismatch:
#    print("Mismatch found!")
#    print(mismatch)
#else:
#    print("ONNX matches PyTorch output")

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

torch_logits_np = torch_node_probs.cpu().numpy().flatten()
onnx_logits_np = onnx_node_probs.cpu().numpy().flatten()
x_in_np = x_in.squeeze(0).cpu().numpy()  # shape: [num_nodes, num_features]

diff_threshold = 0.5

#print("Index,PyTorch_Logit,ONNX_Logit,Difference")
#for i, (pt, ox) in enumerate(zip(torch_logits_np, onnx_logits_np)):
#    diff = abs(pt - ox)
#    if diff > diff_threshold:
#        print(f"{i},{pt:.6f},{ox:.6f},{diff:.6f}")
#        print("Features:")
#        for j, f in enumerate(x_in_np[i]):
#            print(f"  Feature {j}: {f:.4f}")

# Compare outputs
abs_diff = torch.abs(torch_node_probs.cpu() - onnx_node_probs.cpu())
max_diff = abs_diff.max().item()
mean_diff = abs_diff.mean().item()

print(f"ONNX vs PyTorch consistency check:")
print(f"Max difference:  {max_diff:.6e}")
print(f"Mean difference: {mean_diff:.6e}")

print(f"ONNX model saved to {outname}")
