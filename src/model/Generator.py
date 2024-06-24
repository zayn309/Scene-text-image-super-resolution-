import torch
import torch.nn as nn
import torch.optim as optim
import sys
from typing import Tuple
sys.path.append('../')
sys.path.append('./')
from MaxVit import MaxViT
from upscaler import UpscaleTransformModule

class Generator(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 depths: Tuple[int, ...] = (4, 4, 4, 4),
                 channels: Tuple[int, ...] = (128, 128, 128, 128)):
        
        super(Generator, self).__init__()
        
        self.in_channels = in_channels
        self.depth = depths
        self.channels = channels
        self.vit = MaxViT(in_channels=self.in_channels, depths=self.depth, channels=self.channels)
        self.upscaler = UpscaleTransformModule(in_channels=64)
        
    def forward(self, input,tp_features):
        # input should be of shape (B, 3, 16, 64)
        # and output should be of shape (B, 3, 32, 128)

        vit_features = self.vit(input, tp_features)
        upscaled_image = self.upscaler(vit_features)
        return upscaled_image


if __name__ == "__main__":
    # Set up training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    model = Generator().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    print("the training is done on", device)
    print("the model summary: ")
    print(model)

    model.train()
    
    num_epochs = 1000
    batch_size = 1

    # Generate random noise as input and target output
    input_noise = torch.rand(batch_size, 3, 16, 64).to(device)


    for epoch in range(num_epochs):

        # Forward pass
        output = model.vit(input_noise)
        print(output.shape)
        break
        loss = criterion(output, target_noise)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()