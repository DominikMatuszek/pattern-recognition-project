import torch 

# Gets image (B, C, H, W) and returns (B, C', H*2, W*2)
class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels, in_channels, 3, padding=1)
        
        self.activation = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm2d(in_channels)
        
        self.upsample = torch.nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)
    
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
        x = self.upsample(x)
        
        return x     