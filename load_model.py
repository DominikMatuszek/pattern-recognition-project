import torch 
import torchvision
from unet import UNet
from matplotlib import pyplot as plt


def main():
    model = UNet()
    model.load_state_dict(torch.load("m1", map_location=torch.device('cpu')))
    model.eval()
    
    # Load image from images/test/1.jpg
    image = torchvision.io.read_image("images/test/COCO_test2014_000000000001.jpg", mode=torchvision.io.image.ImageReadMode.RGB)
    # Resize to 128x128
    image = torchvision.transforms.functional.resize(image, (128, 128))
    # Normalize
    image = image.float() / 255.0
    # Add batch dimension
    image = image.unsqueeze(0)
    
    pred = model(image)
    
    # Show prediction
    
    pred = pred.squeeze(0)
    
    # Move the channel dimension to the end
    pred = pred.permute(1, 2, 0).detach().numpy()
    
    # Show the base image
    #plt.imshow(image.squeeze(0).permute(1, 2, 0))
    
    plt.imshow(pred, alpha=1)
    
    plt.show()
    
if __name__ == "__main__":
    main()