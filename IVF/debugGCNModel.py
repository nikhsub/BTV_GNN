import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_add, scatter_mean
import numpy as np

# ----------- Custom Edge-aware Conv Layer -------------
class EdgeMLPConv(MessagePassing):
    def __init__(self, in_channels, edge_channels, out_channels):
        super().__init__(aggr='mean')
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        #if not torch.onnx.is_in_onnx_export():
        #    return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        #else:
        #    row, col = edge_index
        #    x_i, x_j = x[row], x[col]
        #    messages = self.mlp(torch.cat([x_i, x_j, edge_attr], dim=1))
        #    return scatter_mean(messages, row, dim=0, dim_size=x.size(0))
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.mlp(torch.cat([x_i, x_j, edge_attr], dim=1))

class SafeLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        count = x.size(-1)
        mean = torch.sum(x, dim=-1, keepdim=True) / count
        var = torch.sum((x - mean) ** 2, dim=-1, keepdim=True) / count
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias

class ONNXScatter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src, index, dim_size):
        output = torch.zeros(dim_size, *src.shape[1:], device=src.device)
        return output.index_add(0, index, src)
        
class GNNModel(torch.nn.Module):
    def __init__(self, indim, outdim, edge_dim):
        super(GNNModel, self).__init__()

        self.bn0 = nn.LayerNorm(indim, eps=1e-6)

        self.onnx_scatter = ONNXScatter()

        self.proj_skip = nn.Linear(indim, outdim)

        self.dummy_token = nn.Parameter(torch.randn(indim)) 

        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(edge_dim, eps=1e-6),
            nn.Linear(edge_dim, outdim),
            nn.ReLU(),
            nn.Linear(outdim, outdim * 2),
            nn.ReLU(),
            nn.Linear(outdim * 2, outdim)
        )

        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, 1),
            nn.Sigmoid()
        )

        self.edge_mlpconv1 = EdgeMLPConv(indim, outdim, outdim//2)
        self.edge_mlpconv2 = EdgeMLPConv(outdim//2, outdim, outdim)

        self.bn1 = nn.LayerNorm(outdim//2, eps=1e-6)
        self.drop1 = nn.Dropout(p=0.3)

        self.bn2 = nn.LayerNorm(outdim, eps=1e-6)
        self.drop2 = nn.Dropout(p=0.2)

        self.gate_layer = nn.Linear(outdim, outdim)

        catout = outdim + outdim

        self.node_pred = nn.Sequential(
            nn.Linear(catout, catout//2),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(catout//2, catout//4),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(catout//4, 1)
        )

        self.layer_outputs = {}
        self.debug_mode = False
    
    def _store_layer_output(self, name, tensor):
        if self.debug_mode:
            self.layer_outputs[name] = tensor.detach().cpu()
        return tensor

    def forward(self, x_in, edge_index, edge_attr):

        if self.debug_mode:
            self.layer_outputs.clear()

        batch_size, num_nodes, _ = x_in.shape
        xf_all, probs_all = [], []

        if edge_index.dtype == torch.float32:
            edge_index = edge_index.long() 

        x = x_in[0]                          # [num_nodes, indim]
        e_idx = edge_index[0]               # [2, num_edges]
        e_attr = edge_attr[0]               # [num_edges, edge_dim]

        self._store_layer_output('input_x', x)
        self._store_layer_output('input_edge_index', e_idx)
        self._store_layer_output('input_edge_attr', e_attr)
        


        #manual_dummy = torch.tensor(
        #    [-2.3952117, 0.54164386, 0.09683848, 0.85531396, 0.7040078,
        #     0.21099302, 1.6392194, 0.24887794, -1.8308424, -1.1111757,
        #     -1.0824525, -1.8598198],
        #    dtype=x.dtype,
        #    device=x.device
        #)
        invalid_mask = (x[:, 0] == -999.0)
        x[invalid_mask] = self.dummy_token
        #x[invalid_mask] = manual_dummy
        self._store_layer_output('after_dummy', x)

        x = self.bn0(x)
        self._store_layer_output('bn0_output', x)

        e_attr_enc = self.edge_encoder(e_attr)
        self._store_layer_output('edge_encoder_output', e_attr_enc)
        
        edge_weights = self.edge_classifier(e_attr).view(-1, 1)
        e_attr_enc = e_attr_enc * edge_weights
        self._store_layer_output('weighted_edge_attr', e_attr_enc)
        
        self._store_layer_output('input_x_beformlp1', x)
        self._store_layer_output('input_edge_index_beforemlp1', e_idx)
        self._store_layer_output('input_edge_attr_enc_beforemlp1', e_attr_enc)

        x1 = self.edge_mlpconv1(x, e_idx, e_attr_enc)
        self._store_layer_output('edge_mlpconv1_output', x1)        

        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1)
        x1 = self.drop1(x1)
        self._store_layer_output('after_mlpconv1_processing', x1)

        x2 = self.edge_mlpconv2(x1, e_idx, e_attr_enc)
        self._store_layer_output('edge_mlpconv2_output', x2)
    
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.drop2(x2)
        self._store_layer_output('after_mlpconv2_processing', x2)

        gate = torch.sigmoid(self.gate_layer(self.proj_skip(x)))
        self._store_layer_output('gate_values', gate)
        #self.last_gate = gate.detach()
        xf = gate * self.proj_skip(x) + torch.sub(1, gate) * x2
        self._store_layer_output('skip_connection_output', xf)

        ones = torch.ones(e_idx.size(1), device=x.device)
        #deg = self.onnx_scatter(ones, e_idx[1], num_nodes).unsqueeze(1).clamp(min=1)
        deg = scatter_add(ones, e_idx[1], dim=0, dim_size=num_nodes).unsqueeze(1).clamp(min=1)
        self._store_layer_output('degree_calculation', deg)
    
        #edge_feats_sum = self.onnx_scatter(e_attr_enc, e_idx[1], num_nodes)
        edge_feats_sum = scatter_add(e_attr_enc, e_idx[1], dim=0, dim_size=num_nodes)
        self._store_layer_output('edge_feats_sum', edge_feats_sum)
    
        edge_feats_mean = edge_feats_sum / deg
        self._store_layer_output('edge_feats_mean', edge_feats_mean)
        
        xf_combined = torch.cat([xf, edge_feats_mean], dim=1)
        self._store_layer_output('combined_features', xf_combined.unsqueeze(0))

        node_probs = self.node_pred(xf_combined)
        self._store_layer_output('final_output', node_probs.unsqueeze(0))

        layer_outputs_stacked = torch.zeros(1)
        layer_names = None
        layer_shapes = None

        if self.debug_mode:
            # Flatten all layer outputs into a single tensor + store metadata
            layer_values = list(self.layer_outputs.values())
            layer_shapes = [tensor.shape for tensor in layer_values]
            layer_names = list(self.layer_outputs.keys())
            
            # Stack all tensors (pad if shapes differ)
            max_len = max(tensor.numel() for tensor in layer_values)
            padded_tensors = [
                torch.nn.functional.pad(
                    tensor.flatten(), 
                    (0, max_len - tensor.numel()), 
                    mode='constant', 
                    value=0
                ) for tensor in layer_values
            ]
            layer_outputs_stacked = torch.stack(padded_tensors)
            
        return xf_combined.unsqueeze(0), node_probs.unsqueeze(0), layer_outputs_stacked, layer_names, layer_shapes
    
    def compare_with_onnx(self, onnx_session, input_dict):
        """Compare PyTorch and ONNX layer outputs"""
        # Run PyTorch in debug mode
        self.debug_mode = True
        self.eval()
        
        torch_inputs = {
        'x_in': torch.from_numpy(input_dict['x_in']).float() if isinstance(input_dict['x_in'], np.ndarray) else input_dict['x_in'],
        'edge_index': torch.from_numpy(input_dict['edge_index']).float() if isinstance(input_dict['edge_index'], np.ndarray) else input_dict['edge_index'],
        'edge_attr': torch.from_numpy(input_dict['edge_attr']).float() if isinstance(input_dict['edge_attr'], np.ndarray) else input_dict['edge_attr']
        }
    
        with torch.no_grad():
            xf_combined, node_probs, layer_outputs_stacked, layer_names, layer_shapes = self(**torch_inputs)
  

        # Prepare ONNX inputs (ensure numpy)
        onnx_inputs = {
        'x_in': torch_inputs['x_in'].cpu().numpy(),
        'edge_index': torch_inputs['edge_index'].cpu().numpy(),
        'edge_attr': torch_inputs['edge_attr'].cpu().numpy()
        }

        onnx_outputs = onnx_session.run(None, onnx_inputs)
        onnx_layer_outputs_stacked = onnx_outputs[2]  # Index 2 = layer_outputs_stacked


        # Reconstruct ONNX dictionary
        onnx_layer_outputs = {}
        for i, (name, shape) in enumerate(zip(layer_names, layer_shapes)):
            # Extract and reshape the tensor
            flat_tensor = onnx_layer_outputs_stacked[i]
            valid_length = np.prod(shape)  # Original tensor size (ignore padding)
            reconstructed_tensor = flat_tensor[:valid_length].reshape(shape)
            onnx_layer_outputs[name] = reconstructed_tensor
        
        
        # Compare layer by layer
        comparison = {}
        for layer_name, pt_tensor in self.layer_outputs.items():
            onnx_tensor = onnx_layer_outputs[layer_name]
            diff = (pt_tensor - onnx_tensor).abs()
            comparison[layer_name] = {
                'max_diff': diff.max().item(),
                'mean_diff': diff.mean().item(),
                'pt_shape': tuple(pt_tensor.shape),
                'onnx_shape': tuple(onnx_tensor.shape)
            }
        
        self.debug_mode = False
        return comparison

    
