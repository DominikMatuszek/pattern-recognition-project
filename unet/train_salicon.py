import torch 
from decoder import DecoderBlock
from encoder import EncoderBlock
from unet import UNet
from salicon_dataset import Salicon
from evaluate_salicon import evaluate_salicon_model
from tqdm import tqdm 

def main():
    epochs = 10

    ds = Salicon("../images/train", "../train", on_gpu=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)

    eval_ds = Salicon("../images/val", "../val", on_gpu=True)
    eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=64, shuffle=True, drop_last=True)

    model = UNet(in_channels=3, out_channels=1)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = torch.nn.BCELoss()

    epoch_train_losses = []
    epoch_eval_losses = []

    for epoch in range(epochs):
        total_loss = 0.0

        for image, mask in tqdm(loader, desc="Epoch " + str(epoch), total=len(loader)):
            optimizer.zero_grad()

            prediction = model(image)
            loss_value = loss(prediction, mask)

            loss_value.backward()
            optimizer.step()

            total_loss += loss_value.item()
        
        avg_loss = total_loss / len(loader)
        epoch_train_losses.append(avg_loss)
        eval_loss = evaluate_salicon_model(model, eval_loader)
        epoch_eval_losses.append(eval_loss)

        print("Epoch", epoch, "loss", avg_loss)
        print("Validation loss", eval_loss)
    
    torch.save(model.state_dict(), "salicon_model.pt")

    from matplotlib import pyplot as plt

    plt.plot(epoch_train_losses, label="Train loss")
    plt.plot(epoch_eval_losses, label="Validation loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()