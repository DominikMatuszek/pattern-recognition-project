from mae_interpolator import InterpolatingMAE
import torch
import timm
from matplotlib import pyplot as plt

from saliency_dataset import SaliencyDataset

def main():
    base = timm.create_model('deit3_large_patch16_224.fb_in1k', pretrained=True)

    for block in base.blocks:
        block.attn.fused_attn = False

    ds = SaliencyDataset("val/", base)

    base_mae = timm.create_model('vit_base_patch16_224.mae', pretrained=True)
    
    model = InterpolatingMAE(mae_out_dim=768, out_dim=ds.get_mask_shape()[0], model=base_mae)
    model.load_state_dict(state_dict=torch.load("mae_interpolator7-epoch0.pt"))

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(base))
    print(count_parameters(model))
    #exit()

    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)

    for i, (img, mask) in enumerate(loader):
        out = model(img)

        figs, axs = plt.subplots(1, 3)

        img = img.squeeze(0).permute(1, 2, 0).numpy()
        mask = mask.squeeze(0).numpy()
        out = out.squeeze(0).detach().numpy()

        axs[0].imshow(img)
        axs[1].imshow(mask)
        axs[2].imshow(out)

        plt.show()

if __name__ == "__main__":
    main()

