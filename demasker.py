import torch

from unet import UNet
from dataset import SaliconMaskedAutoEncoder
from tqdm import tqdm

def main():
    
    model = UNet()
    ds = SaliconMaskedAutoEncoder()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()
    
    
    for epoch in range(10):
        general_loss = 0.0
        
        for image, masked in tqdm(loader):
            optimizer.zero_grad()
            
            pred = model(masked)
            
            loss = loss_fn(pred, image)
            
            loss.backward()
            
            optimizer.step()
            
            general_loss += loss.item()
        
        print("Epoch", epoch, "loss", general_loss / len(loader))
    
    torch.save(model.state_dict(), "m2")
    
if __name__ == "__main__":
    main()
