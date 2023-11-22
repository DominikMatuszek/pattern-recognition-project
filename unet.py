import torch 
import torchvision

from dataset import Salicon  
from tqdm import tqdm    

# Gets image (B, C, H, W) and returns (B, C', H/2, W/2)
class EncoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
                
        self.convolutions = [
            torch.nn.Conv2d(in_channels, in_channels, 3, padding=1),
            torch.nn.Conv2d(in_channels, in_channels, 3, padding=1),
            torch.nn.Conv2d(in_channels, in_channels, 3, padding=1),
            torch.nn.Conv2d(in_channels, in_channels, 3, padding=1),
        ]
        
        self.activation = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm2d(in_channels)
        
        self.downsample = torch.nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        for conv in self.convolutions:
            skip = x
            x = conv(x)
            x += skip
            x = self.activation(x)
        
        x = self.bn(x)
        x = self.downsample(x)
        
        return x
    
    def cuda(self):
        super(EncoderBlock, self).cuda()
        for conv in self.convolutions:
            conv.cuda()
        self.activation.cuda()
        self.bn.cuda()
        self.downsample.cuda()
        
        return self

# Gets image (B, C, H, W) and returns (B, C', H*2, W*2)
class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        self.convolutions = [
            torch.nn.Conv2d(in_channels, in_channels, 3, padding=1),
            torch.nn.Conv2d(in_channels, in_channels, 3, padding=1),
            torch.nn.Conv2d(in_channels, in_channels, 3, padding=1),
            torch.nn.Conv2d(in_channels, in_channels, 3, padding=1),
        ]
        
        self.activation = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm2d(in_channels)
        
        self.upsample = torch.nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)
    
    def forward(self, x):
        for conv in self.convolutions:
            skip = x
            x = conv(x)
            x += skip
            x = self.activation(x)
        
        x = self.bn(x)
        x = self.upsample(x)
        
        return x
    
    def cuda(self):
        super(DecoderBlock, self).cuda()
        for conv in self.convolutions:
            conv.cuda()
        self.activation.cuda()
        self.bn.cuda()
        self.upsample.cuda()
        
        return self

class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.encoders = [
            EncoderBlock(3, 8),
            EncoderBlock(8, 16),
            EncoderBlock(16, 16)
        ]
        
        self.decoders = [
            DecoderBlock(16, 16),
            DecoderBlock(32, 8),
            DecoderBlock(16, 3)
        ]
        
        self.final_activation = torch.nn.Sigmoid()
        self.final_convolution = torch.nn.Conv2d(3, 1, 1)
        
    def forward(self, x):
        skips = []
        
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            
        # Pop last element
        skips = skips[:-1]
                
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x)
            x = torch.cat([x, skip], dim=1)
            
        # Last decoder
        x = self.decoders[-1](x)
        
        x = self.final_convolution(x)
        x = self.final_activation(x)
        
        return x

    def cuda(self):
        super(UNet, self).cuda()
        
        for encoder in self.encoders:
            encoder.cuda()
            
        for decoder in self.decoders:
            decoder.cuda()
        
        self.final_activation.cuda()
        self.final_convolution.cuda()
        
        return self

def main():
    dataset = Salicon()

    model = UNet()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    loss = torch.nn.BCELoss()
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    
    for epoch in range(10):
        general_loss = 0.0
        
        for i, (image, mask) in tqdm(enumerate(data_loader)):
            # Move to GPU
            if torch.cuda.is_available():
                image = image.cuda()
                mask = mask.cuda()
            
            pred = model(image)
            
            loss_value = loss(pred, mask)
            
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
            general_loss += loss_value.item()
            
            #print("Step {} loss {}".format(i, loss_value.item()))
        
        print("Epoch", epoch, "loss", general_loss / len(data_loader))
    
    torch.save(model.state_dict(), "m1")
    
    
if __name__ == "__main__":
    main()