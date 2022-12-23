import torch
import torch.nn as nn

class ECG_CNN(nn.Module):
    def __init__(self, num_classes, kernel_size=5, residual_channels=32, fc_nodes=32):
        super(ECG_CNN, self).__init__()

        self.conv1 = nn.Conv1d(1, residual_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        
        self.residual_block1 = ResidualBlock(residual_channels, residual_channels)
        self.residual_block2 = ResidualBlock(residual_channels, residual_channels)
        self.residual_block3 = ResidualBlock(residual_channels, residual_channels)
        self.residual_block4 = ResidualBlock(residual_channels, residual_channels)
        self.residual_block5 = ResidualBlock(residual_channels, residual_channels)
        
        self.fc1 = nn.Linear(residual_channels, fc_nodes)
        self.fc2 = nn.Linear(fc_nodes, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.residual_block4(x)
        x = self.residual_block5(x)
        x = nn.MaxPool1d(2)(x)
        x = x.squeeze(-1)
        

        x = self.fc1(x)
        x = self.fc2(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, pooling_size=5, stride_pooling=2):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.max_pool = nn.MaxPool1d(pooling_size, stride=stride_pooling)
        
    def forward(self, x):
        skip_connection = torch.clone(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x + skip_connection
        x = self.max_pool(x)
        return x
