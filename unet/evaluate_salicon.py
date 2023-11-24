import torch 
from salicon_dataset import Salicon
from tqdm import tqdm

def evaluate_salicon_model(model, dataset_loader):
    model.eval()
    total_loss = 0.0
    loss = torch.nn.BCELoss()

    with torch.no_grad():
        for image, mask in tqdm(dataset_loader, desc="Evaluating model", total=len(dataset_loader)):
            prediction = model(image)
            loss_value = loss(prediction, mask)
            total_loss += loss_value.item()

    avg_loss = total_loss / len(dataset_loader)
    return avg_loss