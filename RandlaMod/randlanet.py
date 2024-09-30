from buildblock import *
import torch
import torch.nn as nn

class randlanet(nn.Module):
    def __init__(self, in_features, out_features, k):
        super(randlanet, self).__init__()



        self.fc = nn.Linear(in_features, 16)
        
        # ResBlock layers with increasing feature dimensions
        self.resblock1 = Resblock(16, 16, k) 
        self.resblock2 = Resblock(32, 32, k)
        self.resblock3 = Resblock(64, 64, k)
        self.resblock4 = Resblock(128, 128, k) #Change later but right now its (in, 2*out), see resblock

        self.mlp1 = nn.Sequential(
            nn.Linear(256, 512),  
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(256, 128),  
            nn.ReLU()
            #nn.Linear(512, 256)
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.mlp4 = nn.Sequential(
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.mlp5 = nn.Sequential(
            nn.Linear(64, 32),  
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.fc2 = nn.Linear(32, 64)

        self.fc3 = nn.Linear(64,32)

        self.fc4 = nn.Linear(32, out_features)

        self.dropout = nn.Dropout(p=0.5)

    def random_sample(self, features, pos, downsample_factor):
        """
        Randomly samples a subset of points from the feature matrix.
        :param features: [N, d] feature matrix
        :param downsample_factor: int, factor to downsample points by
        :return: [N', d] sampled features
        """
        num_points, feature_dim = features.size()
        num_sampled_points = (num_points // downsample_factor)// 2 * 2
        sampled_indices = torch.randperm(num_points)[:num_sampled_points]
        sampled_features = features[sampled_indices, :]
        sampled_pos = pos[sampled_indices, :]
        return sampled_features, sampled_pos

    def find_nearest_neighbors(self, features):
        """
        Find the nearest neighbor indices for each feature in the feature matrix.
        :param features: [N, d] feature matrix
        :return: indices of nearest neighbors
        """
        norm_features = F.normalize(features, p=2, dim=1)
        pairwise_dists = torch.cdist(norm_features, norm_features)
        pairwise_dists.fill_diagonal_(float('inf'))
        nearest_indices = torch.argmin(pairwise_dists, dim=1)
        return nearest_indices

    def nearest_interpolation(self, features, interp_idx, upsample_factor):
        """
        Upsample features using nearest-neighbor interpolation.
        :param features: [N, d] input feature matrix
        :param interp_idx: [N] nearest neighbor indices
        :param upsample_factor: Factor by which to upsample the feature matrix
        :return: [upsample_factor*N, d] interpolated feature matrix
        """
        num_points, feature_dim = features.size()
        up_num_points = upsample_factor * num_points

        # Initialize the upsampled feature matrix
        upsampled_features = torch.zeros((up_num_points, feature_dim), dtype=features.dtype, device=features.device)
        
        # Place the original features in the upsampled feature matrix
        upsampled_features[::upsample_factor] = features

        # Place the nearest neighbors' features
        for i in range(num_points):
            # Calculate the start and end indices for the upsampled features
            start_idx = i * upsample_factor + 1
            end_idx = start_idx + upsample_factor - 1 
            
            # Get the nearest neighbor index for the current point
            nn_idx = interp_idx[i].item()
            
            # Copy the neighbor's feature into the upsampled feature matrix
            upsampled_features[start_idx:end_idx] = features[nn_idx].repeat(upsample_factor-1, 1)

        return upsampled_features

    def padtensors(self, target_tensor, input_tensor):
        current_size = input_tensor.size(0)
        target_size = target_tensor.size(0)
        if current_size < target_size:
            padding_size = target_size - current_size

            # Randomly sample indices from the existing rows
            random_indices = torch.randint(0, current_size, (padding_size,))

            # Extract the rows using the random indices
            padding_rows = input_tensor[random_indices]

            # Concatenate the padding rows to the original tensor
            input_tensor = torch.cat([input_tensor, padding_rows], dim=0)

        return input_tensor

    def forward(self, x, pos, ei):

        x1_tocat = self.fc(x)
        #print(x1_tocat.shape)

        x2 = self.resblock1(x1_tocat, pos, ei)
        #x2_tocat, pos2 = self.random_sample(x2, pos, downsample_factor=2)
        #print(x2_tocat.shape)

        x3 = self.resblock2(x2, pos, ei)
        #x3_tocat, pos3 = self.random_sample(x3, pos2, downsample_factor=2)
        #print(x3_tocat.shape)

        x4 = self.resblock3(x3, pos, ei)
        #x4_tocat, pos4 = self.random_sample(x4, pos3,  downsample_factor=2)
        #print(x4_tocat.shape)

        x5 = self.resblock4(x4, pos, ei)
        #x5, pos5 = self.random_sample(x5, pos4, downsample_factor=2)
        #print(x5.shape)

        x6 = self.mlp1(x5)
        #print(x6.shape)

        #nearest_indices_x6 = self.find_nearest_neighbors(x6)
        #x7 = self.nearest_interpolation(x6, nearest_indices_x6, 2)
        x7_tocomb = self.mlp2(x6)
        #print(x7_tocomb.shape)
        #x7_tocomb = self.padtensors(x4_tocat, x7_tocomb)
        x7_comb = torch.cat([x4, x7_tocomb], dim=1)

        #nearest_indices_x7comb = self.find_nearest_neighbors(x7_comb)
        #x8 = self.nearest_interpolation(x7_comb, nearest_indices_x7comb, 2)
        x8_tocomb = self.mlp3(x7_comb)
        #x8_tocomb = self.padtensors(x3_tocat, x8_tocomb)
        x8_comb = torch.cat([x3, x8_tocomb], dim=1)

        #nearest_indices_x8comb = self.find_nearest_neighbors(x8_comb)
        #x9 = self.nearest_interpolation(x8_comb, nearest_indices_x8comb, 2)
        x9_tocomb = self.mlp4(x8_comb)
        #x9_tocomb = self.padtensors(x2_tocat, x9_tocomb)
        x9_comb = torch.cat([x2, x9_tocomb], dim=1)

        #nearest_indices_x9comb = self.find_nearest_neighbors(x9_comb)
        #x10 = self.nearest_interpolation(x9_comb, nearest_indices_x9comb, 2)
        x10_tocomb = self.mlp5(x9_comb)
        #x10_tocomb = self.padtensors(x1_tocat, x10_tocomb)
        x10_comb = torch.cat([x1_tocat, x10_tocomb], dim=1)

        x11 = self.fc2(x10_comb)

        x12 = self.fc3(x11)

        x12 = self.dropout(x12)

        output = self.fc4(x12)

        return torch.sigmoid(output)
