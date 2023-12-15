import torch
import torchvision

from PIL import Image

from vit_rollout import VITAttentionRollout
import os 


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, path, transforms = None):
        self.path = path
        self.transforms = transforms
        self.files = os.listdir(path)    

        if transforms is None:
            self.transforms = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.path + self.files[idx])
        img = img.resize((224, 224))

        # Convert to RGB
        img = img.convert('RGB')

        # Apply transforms
        img = self.transforms(img)

        return img 


    

def main():
    ds = ImageDataset("val/")
    
    print(ds[0].shape)

if __name__ == "__main__":
    main()