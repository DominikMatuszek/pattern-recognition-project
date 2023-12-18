import torchvision as tv
from torch.utils.data import DataLoader
import torch

def train(student, epochs, rollout, optimizer, batch_size, loss_fn, **kwargs):
    transform = tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
            tv.transforms.Resize((224,224))
        ]
    )
    dataset = tv.datasets.CIFAR100(
        "./data", train=True, download=True, transform=transform
    )

    loader = DataLoader(dataset, batch_size=batch_size)

    for _ in range(epochs):
        for data, _ in loader:
            teacher_out = rollout(data)
            student_out = student(data)
            
            optimizer.zero_grad()
            loss = loss_fn(teacher_out, student_out)
            loss.backward()
            optimizer.step()
    torch.save(student, f"student_{kwargs['experiment_name']}.pth")