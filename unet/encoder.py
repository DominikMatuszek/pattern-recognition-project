import torch

class EncoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
                
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels, in_channels, 3, padding=1)
        
        
        self.activation = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm2d(in_channels)
        
        self.downsample = torch.nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        skip = x
        x = self.conv1(x)
        x = self.activation(x)
        x = x + skip
        
        skip = x
        x = self.conv2(x)
        x = self.activation(x)
        x = x + skip
        
        skip = x
        x = self.conv3(x)
        x = self.activation(x)
        x = x + skip
        
        skip = x
        x = self.conv4(x)
        x = self.activation(x)
        x = x + skip
        
        x = self.bn(x)
        x = self.downsample(x)
        
        return x