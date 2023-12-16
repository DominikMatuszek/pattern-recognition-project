from student_transformer import StudentTransformer

import torch
import timm
from matplotlib import pyplot as plt

from image_dataset import ImageDataset
from vit_rollout import VITAttentionRollout

from config import CONFIG

def main():
    teacher = timm.create_model(CONFIG["teacher_model"], pretrained=True)

    for block in teacher.blocks:
        block.attn.fused_attn = False

    ds = ImageDataset(CONFIG["dataset_path"])

    base_student = timm.create_model(CONFIG["student_model"], pretrained=True)
    
    model = StudentTransformer(base_student, base_out_dim=CONFIG["student_out_dim"], out_height=14, out_width=14)
    model.load_state_dict(state_dict=torch.load("student0-epoch0.pt"))

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Teacher params:", count_parameters(teacher))
    print("Student params:", count_parameters(model))

    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)

    rollout = VITAttentionRollout(teacher, head_fusion="mean", discard_ratio=0.95)

    for i, img in enumerate(loader):
        out = model(img)
        mask = rollout(img)

        figs, axs = plt.subplots(1, 3)

        img = img.squeeze(0).permute(1, 2, 0).numpy()
        out = out.squeeze(0).detach().numpy()
        mask = mask.squeeze(0).detach().numpy()

        print(img.shape)
        print(out.shape)


        print(mask.shape)

        axs[0].imshow(img)
        axs[1].imshow(mask)
        axs[2].imshow(out)

        plt.show()

if __name__ == "__main__":
    main()

