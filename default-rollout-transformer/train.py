import torch 
import timm 

from mae_interpolator import InterpolatingMAE
from saliency_dataset import SaliencyDataset

from tqdm import tqdm

def main():  
    vit_teacher = timm.create_model('deit3_large_patch16_224.fb_in1k', pretrained=True)

    for block in vit_teacher.blocks:
        block.attn.fused_attn = False

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dataset = SaliencyDataset("val/", vit_teacher, device=device)
    
    base_model = timm.create_model('vit_base_patch16_224.mae', pretrained=True)
    model = InterpolatingMAE(base_model, 768, dataset.get_mask_shape()[0])

    model = model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = torch.nn.BCELoss()

    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)


    for epoch in range(3):
        sum_loss = 0.0

        for i, (img, mask) in tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}"):
            optimiser.zero_grad()

            out = model(img)

            # Flatten the mask and out
            mask = mask.reshape(mask.shape[0], -1)
            out = out.reshape(out.shape[0], -1)

            loss_val = loss(out, mask)
            loss_val.backward()

            optimiser.step()

            sum_loss += loss_val.item()

            if i % 100 == 1:
                print("Loss: {}".format(sum_loss / (i+1)))
            
        torch.save(model.state_dict(), f"mae_interpolator7-epoch{epoch}.pt")



if __name__ == "__main__":
    main()