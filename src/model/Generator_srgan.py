from torch import nn as nn
import torch


class ConvBlock(nn.Module):
    def _init_(self,
                 in_channels,
                 out_channels,
                 discriminator=False,
                 use_act=True,
                 use_bn=True,
                 **kwargs):
        super(ConvBlock,self)._init_()
        self.use_act = use_act
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True) if discriminator else nn.PReLU(num_parameters=out_channels)

    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.cnn(x)


class UpsampleBlock(nn.Module):
    def _init_(self, in_c, scale_factor):
        super(UpsampleBlock,self)._init_()
        self.conv = nn.Conv2d(in_c, in_c * scale_factor ** 2, 3, 1, 1)
        self.ps = nn.PixelShuffle(scale_factor)
        self.act = nn.PReLU(num_parameters=in_c)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


class ResidualBlock(nn.Module):
    def _init_(self, in_channels):
        super(ResidualBlock,self)._init_()
        self.block1 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.block2 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, use_act=False)
        self.conv1x1 = nn.Conv2d(in_channels + 32, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, tp_features):
        out = self.block1(x)
        print(out.shape)
        out = self.block2(out)
        out = torch.cat((out, tp_features), dim=1)
        out = self.conv1x1(out)
        return x + out


class Generator(nn.Module):
    def _init_(self, in_channels=3, num_channels=64, num_blocks=16):
        super(Generator,self)._init_()
        self.ini = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residuals = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=False)
        self.upsamples = nn.Sequential(UpsampleBlock(num_channels, scale_factor=2))
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x, tp_features):
        initial = self.ini(x)
        out = initial
        for residual in self.residuals:
            out = residual(out, tp_features)
        out = self.convblock(out) + initial
        out = self.upsamples(out)
        out = torch.tanh(self.final(out))
        return out



def test():
    low_resolution = 24
    with torch.cuda.amp.autocast():
        x = torch.randn((64, 3, 16, 64))
        tp_features = torch.randn(64, 32, 16, 64)
        gen = Generator()
        gen_out = gen(x, tp_features)
        print(gen_out.shape)

