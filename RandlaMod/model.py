import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNN(torch.nn.Module):
    def __init__(self, num_features, num_output):
        super(GNN, self).__init__()
        
        self.conv1 = GCNConv(num_features, 64, aggr='mean')
        self.conv2 = GCNConv(64, 128, aggr='mean')
        self.conv3 = GCNConv(128, 256, aggr='mean')
        self.conv4 = GCNConv(256, 512, aggr='mean')
        
        
        self.output = GCNConv(512, num_output)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = self.output(x, edge_index)
        
        return F.sigmoid(x)


class MulticlassBCELoss(torch.nn.Module):
    def __init__(self):
        super(MulticlassBCELoss, self).__init__()
    
    def forward(self, output, target):
        loss = F.binary_cross_entropy(output, target, reduction='sum')
        return loss
