import torch 
import timm 
import torchvision

from student_transformer import StudentTransformer
from image_dataset import ImageDataset

from tqdm import tqdm

from vit_rollout import VITAttentionRollout

from config import CONFIG

def training_loop(student, teacher, epochs, rollout, optimiser, loss, loader, device, ft=False):
    for epoch in range(epochs):
        sum_loss = 0.0

        i = 0

        tqdm_bar = tqdm(loader, total=len(loader))

        for i, imgs in enumerate(tqdm_bar):
            imgs = imgs.to(device)

            with torch.no_grad():
                teacher_rollouts = rollout(imgs)

            if CONFIG["downsampling_strategy"] == "DOWNSIZE":
                new_shape = (
                    imgs.shape[2]//CONFIG["downsampling_factor"], # H
                    imgs.shape[3]//CONFIG["downsampling_factor"] # W
                )

                imgs = torch.nn.functional.interpolate(imgs, size=new_shape, mode="bilinear")
                imgs = torch.nn.functional.interpolate(imgs, size=(224, 224), mode="bilinear")

            student_rollout = student(imgs)
            student_rollout = student_rollout.to("cpu")

            optimiser.zero_grad()

            l = loss(student_rollout, teacher_rollouts)
            l.backward()

            optimiser.step()

            sum_loss += l.item()   

            tqdm_bar.set_description(f"Epoch {epoch}; loss {sum_loss/(i+1)}")

        if ft:
            torch.save(student.state_dict(), f"student-{CONFIG['student_model']}-epoch{epoch}-ft.pt")
        else:
            torch.save(student.state_dict(), f"student-{CONFIG['student_model']}-epoch{epoch}.pt")

def main():  
    teacher = timm.create_model(CONFIG["teacher_model"], pretrained=True)

    for block in teacher.blocks:
        block.attn.fused_attn = False

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),

        # Augmentations
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(90),

        torchvision.transforms.RandomPerspective(),
        torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        
        # Translations
        torchvision.transforms.RandomAffine(0, translate=(0.1, 0.1)),
    ])

    dataset = ImageDataset(CONFIG["dataset_path"], transforms=transforms)
    
    base_model = timm.create_model(CONFIG["student_model"], pretrained=True)
    student = StudentTransformer(base_model, CONFIG["student_out_dim"], 14, 14)

    teacher = teacher.to(device)
    student = student.to(device)

    optimiser = torch.optim.Adam(student.parameters(), lr=CONFIG["lr"])
    loss = torch.nn.BCELoss()

    loader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, pin_memory=True)

    rollout = VITAttentionRollout(teacher, head_fusion="mean", discard_ratio=0.95, device=device)

    epochs = CONFIG["linear_layer_epochs"]

    # Train the linear layer
    student.finetuning = False 
    training_loop(student, teacher, epochs, rollout, optimiser, loss, loader, device)
    
    # Load student from checkpoint
    # student.load_state_dict(torch.load("student-deit_tiny_distilled_patch16_224-epoch0.pt"))

    # Finetune the whole model
    student.finetuning = True
    epochs = CONFIG["ft_epochs"]
    optimiser = torch.optim.Adam(student.parameters(), lr=CONFIG["lr"]/10)
    training_loop(student, teacher, epochs, rollout, optimiser, loss, loader, device, ft=True)


if __name__ == "__main__":
    main()