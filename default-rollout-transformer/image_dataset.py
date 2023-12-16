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
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),

        # Augmentations
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(90),

        torchvision.transforms.RandomPerspective(),
        torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        
        # Translations
        torchvision.transforms.RandomAffine(0, translate=(0.1, 0.1)),
    ])

    ds = ImageDataset("val/", transforms=transforms)

    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)

    from matplotlib import pyplot as plt

    for img in loader:
        img = img.squeeze(0).permute(1, 2, 0).numpy()
        print(img.shape)
        plt.imshow(img)
        plt.show()
    

if __name__ == "__main__":
    main()