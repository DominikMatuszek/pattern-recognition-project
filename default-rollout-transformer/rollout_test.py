import torch
import timm

from image_dataset import ImageDataset

from vit_rollout import VITAttentionRollout
from old_rollout import VITAttentionRolloutLegacy

from matplotlib import pyplot as plt

from config import CONFIG

def main():
    ds = ImageDataset(CONFIG["dataset_path"])

    model = timm.create_model(CONFIG["teacher_model"], pretrained=True)

    for block in model.blocks:
        block.attn.fused_attn = False

    model = model.cuda()

    rollout = VITAttentionRollout(model, head_fusion="mean", discard_ratio=0.95)
    old_rollout = VITAttentionRolloutLegacy(model, head_fusion="mean", discard_ratio=0.95)

    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, pin_memory=True)

    for imgs in loader:
        imgs = imgs.to("cuda")

        mask = rollout(imgs)
        old_mask_1 = old_rollout(imgs[0].unsqueeze(0))
        old_mask_2 = old_rollout(imgs[1].unsqueeze(0))

        mask = mask.cpu().numpy()
        imgs = imgs.cpu().numpy()

        # Show first two images, their masks and their legacy masks
        fig, axs = plt.subplots(2, 3)

        axs[0, 0].imshow(imgs[0].transpose(1, 2, 0))
        axs[0, 1].imshow(mask[0])
        axs[0, 2].imshow(old_mask_1)

        axs[1, 0].imshow(imgs[1].transpose(1, 2, 0))
        axs[1, 1].imshow(mask[1])
        axs[1, 2].imshow(old_mask_2)

        plt.show()


if __name__ == "__main__":
    main()