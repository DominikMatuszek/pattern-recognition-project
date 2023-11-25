import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision

class NET(nn.Module):
    def __init__(self): 
        super(NET, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)


        self.aspp_branches = nn.ModuleList([
            nn.Conv2d(512, 256, kernel_size=1, padding=0),
            nn.Conv2d(512, 256, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(512, 256, kernel_size=3, padding=12, dilation=12),
            nn.Conv2d(512, 256, kernel_size=3, padding=18, dilation=18)
        ])


        self.aspp_conv = nn.Conv2d(256*4, 256, kernel_size=1, padding=0)  



        # decoder 

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_dec1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_dec2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_dec3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)


    def forward(self, x):
        saved_activateion = {}

        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, 2, stride=2)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, 2, stride=2)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.max_pool2d(x, 2, stride=2)
        saved_activateion['max_pool_3'] = x.clone()

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=1, padding=1) 
        saved_activateion['max_pool_4'] = x.clone()


        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=1, padding=1) 
        saved_activateion['max_pool_5'] = x.clone()


        aspp_outs = [branch(x) for branch in self.aspp_branches]
        aspp_output = torch.cat(aspp_outs, dim=1)
        aspp_output = F.relu(self.aspp_conv(aspp_output)) 

        saved_activateion['aspp_output'] = aspp_output  


        x = self.upsample(aspp_output)
        x = F.relu(self.conv_dec1(x))

        x = self.upsample(x)
        x = F.relu(self.conv_dec2(x))

        x = self.upsample(x)
        x = F.relu(self.conv_dec3(x))

        x = self.final_conv(x)

        return x 


