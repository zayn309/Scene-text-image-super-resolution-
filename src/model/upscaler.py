import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class UpscaleTransformModule(nn.Module):
    def __init__(self, in_channels):
        super(UpscaleTransformModule, self).__init__()
        # Define initial transformations
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.act1 = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(0.5)
        
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.act2 = nn.LeakyReLU(0.2)
        self.dropout2 = nn.Dropout(0.5)
        
        # First upscaling layer using sub-pixel convolution
        self.conv3 = nn.Conv2d(256, 256 * 4, kernel_size=3, padding=1)
        self.ps1 = nn.PixelShuffle(2)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.LeakyReLU(0.2)
        self.dropout3 = nn.Dropout(0.5)
        
        # Additional transformations after first upscaling
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.act4 = nn.LeakyReLU(0.2)
        self.dropout4 = nn.Dropout(0.5)
        
        self.conv5 = nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1)
        self.ps2 = nn.PixelShuffle(2)
        self.bn5 = nn.BatchNorm2d(128)
        self.act5 = nn.LeakyReLU(0.2)
        self.dropout5 = nn.Dropout(0.5)
        
        # Final convolution to adjust the channel size
        self.conv6 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        
        # Final convolution to refine the upscaled output
        self.final_conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor of shape [B, 64, 8, 32].

        Returns:
            output (torch.Tensor): Transformed and upscaled tensor of shape [B, 3, 32, 128].
        """
        # Initial transformations
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        
        # First upscaling step
        x = self.act3(self.bn3(self.ps1(self.conv3(x))))
        x = self.dropout3(x)
        
        # Additional transformations after first upscaling
        x = self.act4(self.bn4(self.conv4(x)))
        x = self.dropout4(x)
        
        # Second upscaling step
        x = self.act5(self.bn5(self.ps2(self.conv5(x))))
        x = self.dropout5(x)
        
        # Adjust the channel size to 3
        x = self.conv6(x)
        
        # Apply final convolution
        x = self.final_conv(x)

        # Apply tanh to bound output between -1 and 1
        x = torch.tanh(x)

        return x

if __name__ == "__main__":
    model = UpscaleTransformModule(in_channels=64)

    input_tensor = torch.randn(2, 64, 8, 32)  # For example, B=2

    output = model(input_tensor)
    print(output.shape)  # Should output: torch.Size([2, 3, 32, 128])
