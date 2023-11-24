import torch 
import torchvision

from dataset import Salicon  
from tqdm import tqdm    

# Gets image (B, C, H, W) and returns (B, C', H/2, W/2)
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

class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.encoder1 = EncoderBlock(3, 4) # out: 128
        self.encoder2 = EncoderBlock(4,4) # out: 64
        self.encoder3 = EncoderBlock(4,4) # out: 32
        self.encoder4 = EncoderBlock(4,4) # out: 16
        self.encoder5 = EncoderBlock(4,4) # out: 8
        self.encoder6 = EncoderBlock(4,4) # out: 4
        
        
        self.decoder1 = DecoderBlock(4, 4)
        self.decoder2 = DecoderBlock(8, 4)
        self.decoder3 = DecoderBlock(8, 4)
        self.decoder4 = DecoderBlock(8, 4)
        self.decoder5 = DecoderBlock(8, 4)
        self.decoder6 = DecoderBlock(8, 4)
        
        self.final_activation = torch.nn.Sigmoid()
        self.final_convolution = torch.nn.Conv2d(4, 1, 1)
        
    def forward(self, x):
        x = self.encoder1(x)
        skip1 = x
        x = self.encoder2(x)
        skip2 = x
        x = self.encoder3(x)
        skip3 = x
        x = self.encoder4(x)
        skip4 = x
        x = self.encoder5(x)
        skip5 = x
        x = self.encoder6(x)

        x = self.decoder1(x)        
        x = torch.cat([x, skip5], dim=1)
        x = self.decoder2(x)
        x = torch.cat([x, skip4], dim=1)
        x = self.decoder3(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.decoder4(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.decoder5(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.decoder6(x)
                        
        x = self.final_convolution(x)
        x = self.final_activation(x)
        
        return x

def main():
    torch.autograd.set_detect_anomaly(True)
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