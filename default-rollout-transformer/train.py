import torch 
import timm 

from student_transformer import StudentTransformer
from image_dataset import ImageDataset

from tqdm import tqdm

from vit_rollout import VITAttentionRollout

from config import CONFIG

def main():  
    teacher = timm.create_model(CONFIG["teacher_model"], pretrained=True)

    for block in teacher.blocks:
        block.attn.fused_attn = False

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dataset = ImageDataset(CONFIG["dataset_path"])
    
    base_model = timm.create_model(CONFIG["student_model"], pretrained=True)
    student = StudentTransformer(base_model, CONFIG["student_out_dim"], 14, 14)

    teacher = teacher.to(device)
    student = student.to(device)

    optimiser = torch.optim.Adam(student.parameters(), lr=CONFIG["lr"])
    loss = torch.nn.BCELoss()

    loader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, pin_memory=True)

    rollout = VITAttentionRollout(teacher, head_fusion="mean", discard_ratio=0.95)

    epochs = CONFIG["epochs"]

    for epoch in range(epochs):
        sum_loss = 0.0

        i = 0

        tqdm_bar = tqdm(loader, total=len(loader))

        for i, imgs in enumerate(tqdm_bar):
            imgs = imgs.to(device)

            with torch.no_grad():
                teacher_rollouts = [torch.Tensor(rollout(img.unsqueeze(0))) for img in imgs]

            teacher_rollouts = torch.stack(teacher_rollouts)

            student_rollout = student(imgs)
            student_rollout = student_rollout.to("cpu")

            optimiser.zero_grad()

            l = loss(student_rollout, teacher_rollouts)
            l.backward()

            optimiser.step()

            sum_loss += l.item()   

            tqdm_bar.set_description(f"Epoch {epoch}; loss {sum_loss/(i+1)}")

            
        torch.save(student.state_dict(), f"student0-epoch{epoch}.pt")



if __name__ == "__main__":
    main()