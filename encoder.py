import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet
from torchvision import transforms
import numpy as np


class MDLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(MDLSTM, self).__init__()
        
        # Each LSTM will be applied in a different direction (up, down, left, right)
        self.lstm_up = nn.LSTM(input_channels, hidden_channels, batch_first=True)
        self.lstm_down = nn.LSTM(input_channels, hidden_channels, batch_first=True)
        self.lstm_left = nn.LSTM(input_channels, hidden_channels, batch_first=True)
        self.lstm_right = nn.LSTM(input_channels, hidden_channels, batch_first=True)
        
    def forward(self, x):
        # x has shape [batch_size, height, width, channels]
        batch_size, height, width, channels = x.size()
        
        # Prepare input for LSTM
        x_up = x.permute(0, 2, 1, 3).contiguous().view(batch_size * width, height, channels)  # [batch_size * width, height, channels]
        x_down = x_up
        x_left = x.permute(0, 1, 3, 2).contiguous().view(batch_size * height, width, channels)  # [batch_size * height, width, channels]
        x_right = x_left
        
        # Apply LSTM in each direction
        out_up, _ = self.lstm_up(x_up)
        out_down, _ = self.lstm_down(x_down)
        out_left, _ = self.lstm_left(x_left)
        out_right, _ = self.lstm_right(x_right)
        
        # Reshape outputs back to the original spatial dimensions
        out_up = out_up.view(batch_size, width, height, -1).permute(0, 2, 1, 3)
        out_down = out_down.view(batch_size, width, height, -1).permute(0, 2, 1, 3)
        out_left = out_left.view(batch_size, height, width, -1).permute(0, 2, 1, 3)
        out_right = out_right.view(batch_size, height, width, -1).permute(0, 2, 1, 3)
        
        # Sum the outputs from all directions
        out = out_up + out_down + out_left + out_right
        return out

class DenseMD(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24), num_init_features=64, 
                 bn_size=4, drop_rate=0, num_classes=256, hidden_channels=256):
        super(DenseMD, self).__init__()
        
        # Calculate final number of features after the last transition
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                num_features = num_features // 2

        # Create base DenseNet
        self.densenet = DenseNet(
            growth_rate=growth_rate,
            block_config=block_config,
            num_init_features=num_init_features,
            bn_size=bn_size,
            drop_rate=drop_rate,
            num_classes=num_classes
        )
        
        # Modify first conv layer
        self.densenet.features.conv0 = nn.Conv2d(1, num_init_features, 
                                                kernel_size=5, stride=2,
                                                padding=2, bias=False)
        
        # Efficient dimension reduction with correct input size
        self.dim_reduction = nn.Sequential(
            nn.Conv2d(num_features, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # MDLSTM and other components
        self.mdlstm = MDLSTM(input_channels=hidden_channels, hidden_channels=hidden_channels)
        self.W_phi = nn.Linear(1, hidden_channels, bias=False)
        self.W_psi = nn.Linear(1, hidden_channels, bias=False)
        self.conv1x1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        self.residual = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)

    def forward(self, x):
        # Extract features through modified DenseNet
        x = self.densenet.features.conv0(x)
        x = self.densenet.features.norm0(x)
        x = self.densenet.features.relu0(x)
        x = self.densenet.features.pool0(x)
        
        # Pass through dense blocks and transitions
        x = self.densenet.features.denseblock1(x)
        x = self.densenet.features.transition1(x)
        x = self.densenet.features.denseblock2(x)
        x = self.densenet.features.transition2(x)
        x = self.densenet.features.denseblock3(x)
        
        # Apply final norm before dimension reduction
        x = F.relu(x, inplace=True)
        x = F.batch_norm(x, None, None, None, None, True, 0.1, 1e-5)
        
        # Rest of the forward pass
        x = self.dim_reduction(x)
        
        # Apply MDLSTM to get contextual features
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        rnn_features = self.mdlstm(x)
        rnn_features = rnn_features.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Add positional embeddings
        batch_size, channels, height, width = rnn_features.size()
        device = rnn_features.device
        
        # Create position encodings more explicitly
        y_coords = torch.arange(height, device=device).float() / max(height - 1, 1)  # [H]
        x_coords = torch.arange(width, device=device).float() / max(width - 1, 1)    # [W]
        
        # Create coordinate matrices
        y_coords = y_coords.view(-1, 1).repeat(1, width)   # [H, W]
        x_coords = x_coords.view(1, -1).repeat(height, 1)  # [H, W]
        
        # Add channel dimension for linear layers
        y_coords = y_coords.unsqueeze(-1)  # [H, W, 1]
        x_coords = x_coords.unsqueeze(-1)  # [H, W, 1]
        
        # Apply position embeddings
        phi_x = self.W_phi(x_coords)  # [H, W, hidden_channels]
        psi_y = self.W_psi(y_coords)  # [H, W, hidden_channels]
        
        # Combine embeddings and reshape to match feature dimensions
        position_embedding = phi_x + psi_y  # [H, W, hidden_channels]
        position_embedding = position_embedding.permute(2, 0, 1)  # [hidden_channels, H, W]
        position_embedding = position_embedding.unsqueeze(0)  # [1, hidden_channels, H, W]
        position_embedding = position_embedding.expand(batch_size, -1, -1, -1)  # [B, hidden_channels, H, W]
        
        # Add position embeddings to features
        final_features = rnn_features + position_embedding
        
        # Apply shallow convolution with residual connection
        output = self.conv1x1(final_features) + self.residual(final_features)
        
        return output
