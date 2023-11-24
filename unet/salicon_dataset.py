import torch
import torchvision
import os 

class Salicon(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, on_gpu=False):
        super(Salicon, self).__init__()

        self.images_dir = images_dir
        self.masks_dir = masks_dir

        if on_gpu: 
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.train_images = os.listdir(self.images_dir)
        self.train_masks = os.listdir(self.masks_dir)
        
    def __len__(self):
        return len(self.train_images)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.train_images[idx])
        mask_path = os.path.join(self.masks_dir, self.train_masks[idx])
        
        image = torchvision.io.read_image(image_path, mode=torchvision.io.image.ImageReadMode.RGB)
        mask = torchvision.io.read_image(mask_path, mode=torchvision.io.image.ImageReadMode.GRAY)

        image = image.to(self.device)
        mask = mask.to(self.device)

        image = image.float() / 255.0
        mask = mask.float() / 255.0
        
        # Resize image and mask
        #image = torchvision.transforms.Resize((400, 400))(image)
        #mask = torchvision.transforms.Resize((400, 400))(mask)

        # Concatenate image and mask so that we can apply the same transformations
        # This is probably the worst way to do this 
        temp = torch.cat((image, mask), dim=0)
        
        temp = torchvision.transforms.RandomCrop((256, 256))(temp)
        
        image = temp[:3, :, :]
        mask = temp[3, :, :]
        
        mask.unsqueeze_(0)
        
        # Adjust colours 
        image = torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)(image)

        # Gaussian
        image = torchvision.transforms.GaussianBlur(kernel_size=5)(image)

        return image, mask      
    
def main():
    from matplotlib import pyplot as plt

    dataset = Salicon("../images/train", "../train")
    
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    for i in range(3):
        image, mask = dataset[i]
        ax[i][0].imshow(image.permute(1, 2, 0))
        ax[i][1].imshow(mask[0])
    
    plt.show()

if __name__ == "__main__":
    main()