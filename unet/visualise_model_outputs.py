from unet import UNet
from salicon_dataset import Salicon
import torch
from matplotlib import pyplot as plt


def main():
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load("salicon_model.pt"))
    model.cuda()
    model.eval()

    ds = Salicon("../images/val", "../val", on_gpu=True)

    # Show 3 images from the validation set

    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(1, 4):
        image, mask = ds[i]
        prediction = model(image.unsqueeze(0))

        image = image.cpu()
        mask = mask.cpu()
        prediction = prediction.cpu()

        ax[i-1][0].imshow(image.permute(1, 2, 0))
        ax[i-1][1].imshow(mask[0])
        ax[i-1][2].imshow(prediction[0][0].detach().numpy())

    plt.show()

if __name__ == "__main__":
    main()