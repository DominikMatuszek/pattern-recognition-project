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
        
        mask.unsqueeze_(0)
        
        return image, mask      
    
class SaliconMaskedAutoEncoder(torch.utils.data.Dataset):
    def __init__(self):
        super(SaliconMaskedAutoEncoder, self).__init__()
        
        self.images_dir = "images/test"        
        self.train_images = os.listdir(self.images_dir)
        
    def __len__(self):
        return len(self.train_images)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.train_images[idx])
        
        image = torchvision.io.read_image(image_path, mode=torchvision.io.image.ImageReadMode.RGB)
        
        # If cuda is available, move to cuda
        if torch.cuda.is_available():
            image = image.cuda()
        
        image = image.float() / 255.0

        # Random crop 
        image = torchvision.transforms.RandomCrop((256, 256))(image)
        
        # Moar augmentation
        image = torchvision.transforms.RandomHorizontalFlip()(image)
        image = torchvision.transforms.RandomVerticalFlip()(image)
        image = torchvision.transforms.RandomRotation(15)(image)
        
        # Mask the image
        patch_width = 16
        
        #mask = torch.cuda.FloatTensor(image.shape[1] // patch_width, image.shape[2] // patch_width).uniform_() > 0.8
        mask = torch.FloatTensor(image.shape[1] // patch_width, image.shape[2] // patch_width).uniform_() > 0.8
        mask = mask.float()
                
        # Add batch dimension
        mask.unsqueeze_(0)
        mask.unsqueeze_(0)
                        
        # Upsample the mask
        mask = torchvision.transforms.functional.resize(mask, (256, 256))
        
        # Pointwise multiplication of the mask and the image
        mask.squeeze_(0)
                
        masked = image * mask
       
        # Color jitter on the masked image
        #masked = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(masked)
       
        return image, masked  

def main():
    dataset = Salicon()
    
    image, mask = dataset[0]
    
    print(image.shape, mask.shape)
    
    torchvision.utils.save_image(image, "test.png")
    torchvision.utils.save_image(mask, "test_mask.png")
    
def main2():
    dataset = SaliconMaskedAutoEncoder()
    
    for image, masked in dataset:
        torchvision.utils.save_image(image, "test.png")
        torchvision.utils.save_image(masked, "test_masked.png")
        break
    
if __name__ == "__main__":
    main2()