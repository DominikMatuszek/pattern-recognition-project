import torch 
import torchvision 
from PIL import Image

from config import IMAGENET_VAL_DIR
import os 

class ImageNet(torch.utils.data.Dataset):
    def __init__(self, dir):
        super(ImageNet, self).__init__()
        self.dir = dir
        self.imgs = os.listdir(dir)
        self.imgs = sorted(self.imgs)
        
        labels = open("val.txt", "r").readlines()
        labels = [int(label.strip().split()[1]) for label in labels]
        self.labels = labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(f"{self.dir}/{self.imgs[idx]}")
        img = img.resize((224, 224))

        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")

        img = torchvision.transforms.ToTensor()(img)
        return (img, self.labels[idx])
    
def main():
    from matplotlib import pyplot as plt

    dataset = ImageNet(IMAGENET_VAL_DIR)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    img, label = dataset[49999]

    plt.imshow(img.permute(1, 2, 0))
    
    # Describe the image on the plot with the label
    plt.title(f"Label: {label}")
    plt.show()

if __name__ == "__main__":
    main()