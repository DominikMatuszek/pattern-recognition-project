import torch
import torchvision

from PIL import Image

from vit_rollout import VITAttentionRollout
import os 


class SaliencyDataset(torch.utils.data.Dataset):
    def __init__(self, path, model, device=torch.device('cpu')):
        self.model = model.to(device)
        self.device = device

        self.path = path

        self.rollout = VITAttentionRollout(model, discard_ratio=0.95)

        self.files = os.listdir(path)    

        self.mask_shape = self[0][1].shape

    def get_mask_shape(self):
        return self.mask_shape

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.path + self.files[idx])
        img = img.resize((224, 224))

        # Convert to RGB
        img = img.convert('RGB')

        # Convert to tensor
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        img = transform(img)

        # To uint8

        img = img * 255
        img = img.to(torch.uint8)

        # Augment
        img = torchvision.transforms.AutoAugment()(img)

        # To float
        img = img.to(torch.float32)
        img = img / 255

        img = img.unsqueeze(0)
        img = img.to(self.device)

        # Get the attention maps
        mask = self.rollout(img)

        mask = torch.from_numpy(mask).to(self.device)

        # Apply the mask
        return img.squeeze(0), mask
    

def main():
    import timm 
    model = timm.create_model('deit3_large_patch16_224.fb_in1k', pretrained=True)

    for block in model.blocks:
        block.attn.fused_attn = False

    dataset = SaliencyDataset("val/", model)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for img, mask in loader:
        print(img.shape, mask.shape)
        print(dataset.get_mask_shape())
        
        from matplotlib import pyplot as plt

        figs, axs = plt.subplots(1, 2)

        mask = mask.squeeze(0)
        img = img.squeeze(0).permute(1, 2, 0).numpy()

        axs[0].imshow(img)
        axs[1].imshow(mask)

        plt.show()
    

if __name__ == "__main__":
    main()