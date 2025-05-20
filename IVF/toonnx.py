import torch
from GCNModel import *
import argparse

parser = argparse.ArgumentParser("GNN testing")

parser.add_argument("-lm",  "--load_model", default="", help="Load model file")
parser.add_argument("-o", "--output", default="testmod", help="Name of output onnx file")
args = parser.parse_args()


trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']
edge_features = ['dca', 'deltaR', 'dca_sig', 'cptopv', 'pvtoPCA_1', 'pvtoPCA_2', 'dotprod_1', 'dotprod_2', 'pair_mom', 'pair_invmass']

model = GNNModel(indim=len(trk_features), outdim=48, edge_dim=len(edge_features))
model.load_state_dict(torch.load(args.load_model, map_location=torch.device('cpu')))
model.eval()

num_nodes = 100  # Example number of nodes
num_edges = 300  # Example number of edges
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Random example inputs
x_in = torch.randn((num_nodes, len(trk_features)), dtype=torch.float32)
edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
edge_attr = torch.randn((num_edges, len(edge_features)), dtype=torch.float32) 

model.to(device)
x_in = x_in.to(device)
edge_index = edge_index.to(device)
edge_attr = edge_attr.to(device)

output = model(x_in, edge_index, edge_attr)
print("Model output shape:", output[0].shape, output[1].shape)

outname = args.output+".onnx"
#onnx_program = torch.onnx.dynamo_export(model, x_in, edge_index)
#onnx_program.save("testmodel.onnx")

torch.onnx.export(
    model,                           # The model to export
    (x_in, edge_index, edge_attr),       # Example inputs
    outname,                    # Output file path
    opset_version=17,                # Specify the ONNX opset version
    input_names=["x_in", "edge_index", "edge_attr"],  # Input names
    output_names=["xf", "node_probs"],   # Output names
    dynamic_axes={                    # Support dynamic graph sizes
        "x_in": {0: "num_nodes"},     # First dimension of x_in is dynamic
        "edge_index": {1: "num_edges"},
        "edge_attr": {0: "num_edges"}  # Second dimension of edge_index is dynamic
    }
)
