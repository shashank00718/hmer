import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet


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
                 bn_size=4, drop_rate=0, num_classes=1000, hidden_channels=64):
        super(DenseMD, self).__init__()
        
        # DenseNet as CNN-based feature extractor
        self.densenet = DenseNet(growth_rate=growth_rate, block_config=block_config,
                                 num_init_features=num_init_features, bn_size=bn_size,
                                 drop_rate=drop_rate, num_classes=num_classes)
        
        # MDLSTM as the RNN-based feature extractor
        self.mdlstm = MDLSTM(input_channels=num_init_features + sum(block_config) * growth_rate, 
                             hidden_channels=hidden_channels)
        
        # Position Embeddings
        self.W_phi = nn.Linear(1, hidden_channels, bias=False)
        self.W_psi = nn.Linear(1, hidden_channels, bias=False)
        
        # Final convolution with residual connection
        self.conv1x1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.residual = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        # Feature extraction through DenseNet
        cnn_features = self.densenet(x)  # Shape: [batch_size, height, width, channels]
        
        # Apply MDLSTM to get contextual features
        rnn_features = self.mdlstm(cnn_features)  # Shape: [batch_size, height, width, channels]
        
        # Add positional embeddings
        batch_size, height, width, channels = rnn_features.size()
        px, py = torch.meshgrid(torch.arange(height), torch.arange(width))
        
        # Normalize position embeddings
        px, py = px.float(), py.float()
        px, py = px / (height - 1), py / (width - 1)
        
        # Position embedding
        phi_x = self.W_phi(px.view(-1, 1)).view(batch_size, height, width, -1)
        psi_y = self.W_psi(py.view(-1, 1)).view(batch_size, height, width, -1)
        
        # Add position embeddings to the feature map
        final_features = rnn_features + phi_x + psi_y
        
        # Apply shallow convolution with residual connection
        conv_out = self.conv1x1(final_features) + self.residual(final_features)
        
        return conv_out
