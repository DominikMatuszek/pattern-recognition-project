from cfg import load_config
from models import trans_student, load_teacher
from train import train
from vit_grad_rollout import VITAttentionGradRollout
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch


def main():
    config = load_config()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((224, 224)),
        ]
    )
    dataset = datasets.CIFAR100(
        "./data", train=True, download=True, transform=transform
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    student_model = trans_student(**config["student"])
    teacher_model = load_teacher(**config["teacher"])
    rollout = VITAttentionGradRollout(teacher_model, discard_ratio=0.5)

    for block in teacher_model.blocks:
        block.attn.fused_attn = False

    optimizer = torch.optim.Adam(student_model.parameters(recurse=True))
    loss_fn = torch.nn.MSELoss()

    train(
        student=student_model,
        teacher=teacher_model,
        rollout=rollout,
        data_loader=loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        batch_size=1,
        **config["training"],
    )


if __name__ == "__main__":
    main()
