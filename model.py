import torch.nn as nn
import torch 
class Conv_ReLU_Block(nn.Module):
    def __init__(self, s):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU(s)
        
    def forward(self, x):
        return self.relu(self.conv(x))

class Net(nn.Module):
    def __init__(self, upscale_factor, d=64, s=32, m=6):
        super(Net, self).__init__()

        self.input = nn.Conv2d(1, d, (5, 5), (1, 1), (2, 2))
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(s)
        )
        self.residual_layer = self.make_layer(Conv_ReLU_Block, s, m)
        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(d)
        )
        self.transform = nn.Conv2d(d, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1)) 
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
    def make_layer(self, block, s, m):
        layers = []
        for _ in range(m):
            layers.append(block(s))
        return nn.Sequential(*layers)
    def forward(self, x):
        residual = x
        out = torch.tanh(self.input(x))
        out = self.shrinking(out)
        out = self.residual_layer(out)
        out = self.expanding(out)
        out = torch.add(out,residual)
        out = torch.sigmoid(self.pixel_shuffle(self.transform(out)))
        return out

if __name__ == "__main__":
    model = Net(upscale_factor=3)
    print(model)
