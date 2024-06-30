import torch.nn as nn 
import torch  
from typing import Tuple
import sys

sys.path.append('../')
sys.path.append('./')
from model.MaxVit_comp import MaxViTStage


class MaxViT(nn.Module):
    """ Implementation of the MaxViT proposed in:
        https://arxiv.org/pdf/2204.01697.pdf

    Args:
        in_channels (int, optional): Number of input channels to the convolutional stem. Default 3
        depths (Tuple[int, ...], optional): Depth of each network stage. Default (2, 2, 5, 2)
        channels (Tuple[int, ...], optional): Number of channels in each network stage. Default (64, 128, 256, 512)
        num_classes (int, optional): Number of classes to be predicted. Default 1000
        embed_dim (int, optional): Embedding dimension of the convolutional stem. Default 64
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        norm_layer_transformer (Type[nn.Module], optional): Normalization layer in Transformer. Default: nn.LayerNorm
        global_pool (str, optional): Global polling type to be utilized. Default "avg"
    """

    def __init__(
            self,
            in_channels: int = 3,
            depths: Tuple[int, ...] = (2, 2, 5, 2),
            channels: Tuple[int, ...] = (64, 128, 256, 512),
            embed_dim: int = 64,
            num_heads: int = 32,
            grid_window_size: Tuple[int, int] = (4, 4),
            attn_drop: float = 0.,
            drop=0.,
            drop_path=0.,
            mlp_ratio=4.,
            act_layer=nn.GELU,
            norm_layer=nn.BatchNorm2d,
            norm_layer_transformer=nn.LayerNorm,
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(MaxViT, self).__init__()
        # Check parameters
        assert len(depths) == len(channels), "For each stage a channel dimension must be given."
        
        # Init convolutional stem
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=(3, 3), stride=(2, 2),
                    padding=(1, 1))
        self.act1 = act_layer()
        self.conv2 = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim * 2, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1))
        self.act2 = act_layer()
        # Init blocks
        drop_path = torch.linspace(0.0, drop_path, sum(depths)).tolist()
        stages = []
        for index, (depth, channel) in enumerate(zip(depths, channels)):
            stages.append(
                MaxViTStage(
                    depth=depth,
                    in_channels= embed_dim * 2 if index == 0 else channels[index - 1],
                    out_channels=channel,
                    num_heads=num_heads,
                    grid_window_size=grid_window_size,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path[sum(depths[:index]):sum(depths[:index + 1])],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    norm_layer_transformer=norm_layer_transformer
                )
            )
        self.stages = nn.ModuleList(stages)
        self.proj = nn.Conv2d(embed_dim * 2 * 4, embed_dim * 2, kernel_size=1)
        self.act3 = act_layer()
        self.conv3 = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1)
        self.act4 = act_layer()
        self.tp_projection = nn.ModuleList([nn.Conv2d((embed_dim * 2) + 32,(embed_dim * 2),kernel_size = 1) for _ in range(depth)])  
        
        
    def forward(self, input: torch.Tensor, TP_features: torch.Tensor) -> torch.Tensor:
        """ Forward pass of feature extraction.

        Args:
            input (torch.Tensor): Input images of the shape [B, C, H, W].

        Returns:
            output (torch.Tensor): Image features of the backbone.
        """
        # Initial convolution and activation
        initial = self.act1(self.conv1(input))
        out = self.act2(self.conv2(initial))
        stages_output = []
        
        for idx, stage in enumerate(self.stages):
            skip_input = out
            out = stage(out)
            
            out = torch.cat((out, TP_features), dim=1)
            out = self.tp_projection[idx](out)
            out += skip_input  # In-place addition
            
            stages_output.append(out)

        # Concatenate outputs of all stages except the last one
        concatenated_output = torch.cat(stages_output, dim=1)
        projected_output = self.proj(concatenated_output)

        output = self.act4(self.conv3(projected_output))  # In-place addition

        return output

def test_MaxVit() -> None:
    model = MaxViT(in_channels=3, depths=[4,4,4,4], channels=[128,128,128,128])
    input = torch.randn(2,3,16,64)
    tp_features = torch.randn(2, 32, 8, 32)
    output = model(input,tp_features)
    print(output.shape)

    
if __name__ == "__main__":
    test_MaxVit()
