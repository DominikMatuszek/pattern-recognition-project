from student_transformer import StudentTransformer

import torch
import timm
from matplotlib import pyplot as plt

from image_dataset import ImageDataset

from config import CONFIG

def main():
    teacher = timm.create_model(CONFIG["teacher_model"], pretrained=True)

    for block in teacher.blocks:
        block.attn.fused_attn = False

    ds = ImageDataset(CONFIG["dataset_path"])

    base_mae = timm.create_model(CONFIG["student_model"], pretrained=True)
    
    model = StudentTransformer(CONFIG["student_out_dim"], out_dim=ds.get_mask_shape()[0], model=base_mae)
    model.load_state_dict(state_dict=torch.load("mae_interpolator7-epoch0.pt"))

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(teacher))
    print(count_parameters(model))

    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)

    for i, img in enumerate(loader):
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

