import torch 
import torchvision

import os 

class Salicon(torch.utils.data.Dataset):
    def __init__(self):
        super(Salicon, self).__init__()
        
        self.images_dir = "images/train"
        self.masks_dir = "masks"
        
        self.train_images = os.listdir(self.images_dir)
        self.train_masks = os.listdir(self.masks_dir)
        
    def __len__(self):
        return len(self.train_images)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.train_images[idx])
        mask_path = os.path.join(self.masks_dir, self.train_masks[idx])
        
        image = torchvision.io.read_image(image_path, mode=torchvision.io.image.ImageReadMode.RGB)
        mask = torchvision.io.read_image(mask_path, mode=torchvision.io.image.ImageReadMode.GRAY)
        
        image = image.float() / 255.0
        mask = mask.float() / 255.0
        
        # Random crop -- concatenate image and mask
        temp = torch.cat((image, mask), dim=0)
        temp = torchvision.transforms.RandomCrop((256, 256))(temp)
        
        # Random horizontal flip
        temp = torchvision.transforms.RandomHorizontalFlip()(temp)
        
        # Random vertical flip
        temp = torchvision.transforms.RandomVerticalFlip()(temp)
        
        # Random rotation
        temp = torchvision.transforms.RandomRotation(15)(temp)
        
        image = temp[:3, :, :]
        mask = temp[3, :, :]
        
        
        return image, mask      
    
def main():
    dataset = Salicon()
    
    image, mask = dataset[0]
    
    print(image.shape, mask.shape)
    
    torchvision.utils.save_image(image, "test.png")
    torchvision.utils.save_image(mask, "test_mask.png")
if __name__ == "__main__":
    main()