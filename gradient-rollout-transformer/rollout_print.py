from matplotlib import pyplot as plt
def example(loader, n, sal_fn):
    for img,_ in zip(loader):
        figs, axe = plt.subplots(1,2)
        saliency = sal_fn(img)
        img = img.squeeze().permute(1,2,0)
        # saliency.squeeze(0).detach().numpy()

        axe[0].imshow(saliency)
        axe[1].imshow(img)
        
        plt.show()
        break

    