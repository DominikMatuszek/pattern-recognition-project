import torch
import torchvision
from torchvision import transforms
import timm
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Subset
import numpy as np

import matplotlib.pyplot as plt
from vit_grad_rollout import VITAttentionGradRollout


class TeacherStudentModel(nn.Module):
    def __init__(self, teacher, student):
        super(TeacherStudentModel, self).__init__()
        self.student = student
        self.teacher = teacher
        self.attention_rollout = VITAttentionGradRollout(teacher, discard_ratio=0.95)
        
    def forward(self, x, labels):
        x.requires_grad = True

        input_student = transforms.functional.resize(x, (70,70), antialias=True)
        input_student = transforms.functional.resize(input_student, (224,224), antialias=True)

        target = self.attention_rollout(x, labels)
        output = self.student(input_student)

        return output, target

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import torchvision.transforms as transforms
    from torchvision.datasets import CIFAR100
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    cifar100_train = CIFAR100(root='./data', train=True, download=True, transform=transform)
    cifar100_test = CIFAR100(root='./data', train=False, download=True, transform=transform)

    train_dataloader = DataLoader(cifar100_train, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(cifar100_test, batch_size=32, shuffle=False)
    subset_size = int(0.5 * len(cifar100_train))
    subset_indices = np.random.choice(len(cifar100_train), subset_size, replace=False)

    cifar100_train = Subset(cifar100_train, subset_indices)
    train_dataloader = DataLoader(cifar100_train, batch_size=16, shuffle=True)


    model_teacher = timm.create_model('deit3_large_patch16_224.fb_in1k', pretrained=True)
    for param in model_teacher.parameters():
        param.requires_grad = True

    for block in model_teacher.blocks:
        block.attn.fused_attn = False


    model_teacher.to(device)
    model_teacher.eval()    


    model_student = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
    num_ftrs = model_student.classifier.in_features
    model_student.classifier = nn.Linear(num_ftrs, 400)

    for param in model_student.parameters():
        param.requires_grad = True   


    additional_layers = nn.Sequential(
    nn.ReLU(),
    nn.Linear(400, 300),
    nn.ReLU(),
    nn.Linear(300, 196),
    nn.Sigmoid()
    )


    model_student = nn.Sequential(
        model_student.features,
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        model_student.classifier,
        additional_layers
    )

    model_student = model_student.to(device)


    model = TeacherStudentModel(model_teacher, model_student)
    model = model.to(device)


    optimizer = torch.optim.Adam(model_student.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss().to(device)

    losses = []
    steps = []


    torch.cuda.empty_cache()

    for epoch in range(10):
        print("EPOCH: ", epoch+1)
        for i, (images, labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output, target = model(images, labels)
            output = output.reshape(14,14)
    
            print("Output shape:", output.shape)
            print("Target shape:", target.shape)
    
    
            loss = criterion(target, output)
            loss.backward()
            optimizer.step()
    
            if (i+1) % 500 == 0:
                losses.append(loss.item())
                steps.append(epoch * len(train_dataloader) + i + 1)
    
            if (i+1) % 10000 == 0:
                print(f"STEP: {i + 1}, loss: {loss.item()}")
                fig, axes = plt.subplots(1, 3, figsize=(10, 5))
                axes[0].imshow(images[0].permute(1, 2, 0).cpu().detach())
                axes[1].imshow(target[0].cpu().detach())
                axes[2].imshow(output[0].cpu().detach())
                plt.savefig(f"train{epoch}_{i}.png")
                plt.close(fig)
    
        plt.plot(steps, losses)
        plt.title(f'Loss over time (after {epoch + 1} epoch)')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.savefig(f'Loss{epoch + 1}.png')
        plt.close()
    
        torch.save(model.student.state_dict(), 'model_student_state.pth')
    