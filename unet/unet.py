import torch 
from decoder import DecoderBlock
from encoder import EncoderBlock
from convolutional_attention import ConvolutionalAttention

class UNet(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        self.encoder1 = EncoderBlock(in_channels, 4) # out: 128
        self.encoder2 = EncoderBlock(4,8) # out: 64
        self.encoder3 = EncoderBlock(8,16) # out: 32
        self.encoder4 = EncoderBlock(16,32) # out: 16
        self.encoder5 = EncoderBlock(32,64) # out: 8
        self.encoder6 = EncoderBlock(64,128) # out: 4
        
        
        self.decoder1 = DecoderBlock(128, 64)
        self.decoder2 = DecoderBlock(128, 32)
        self.decoder3 = DecoderBlock(64, 16)
        self.decoder4 = DecoderBlock(32, 8)
        self.decoder5 = DecoderBlock(16, 4)
        self.decoder6 = DecoderBlock(8, 4)
        
        self.att1 = ConvolutionalAttention(128)
        self.att2 = ConvolutionalAttention(64)
        self.att3 = ConvolutionalAttention(32)
        self.att4 = ConvolutionalAttention(16)
        self.att5 = ConvolutionalAttention(8)


        self.final_activation = torch.nn.Sigmoid()
        self.final_convolution = torch.nn.Conv2d(4, out_channels, 1)
        
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
        x = self.att1((x, x, x))

        x = self.decoder2(x)
        x = torch.cat([x, skip4], dim=1)
        x = self.att2((x, x, x))

        x = self.decoder3(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.att3((x, x, x))

        x = self.decoder4(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.att4((x, x, x))

        x = self.decoder5(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.att5((x, x, x))

        x = self.decoder6(x)
                        
        x = self.final_convolution(x)
        x = self.final_activation(x)
        
        return x