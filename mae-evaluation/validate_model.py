import torch 
import torchvision

from imagenet_validation import ImageNet

from config import IMAGENET_VAL_DIR
from tqdm import tqdm 

class Validator:
    def __init__(self, log_file_dir):
        self.log_file_dir = log_file_dir
        self.log_file = open(log_file_dir, "w")

    def __calculate_accuracy_for_model(self, model, device):
        batch_size = 512
        dataset = ImageNet(IMAGENET_VAL_DIR)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=12)

        model = model.to(device)

        correct = 0

        with torch.no_grad():
            bar = tqdm(enumerate(dataloader), total=len(dataloader))
            for i, (img, label) in bar:       
                img = img.to(device)
                label = label.to(device)

                pred = model(img)

                correct += (pred.argmax(dim=1) == label).sum().item()
                #print("Accuracy: ", correct / ((i+1)*batch_size))
                bar.set_description(f"Avg acc: {correct /((i+1) * batch_size)}")

        return correct / (len(dataloader)  * batch_size)

    def log_accuracy_for_model(self, model, device, model_name):
        print("Evaluating", model_name)
        with torch.no_grad():
            accuracy = self.__calculate_accuracy_for_model(model, device)
        self.log_file.write(f"{model_name}: {accuracy}\n")
        self.log_file.flush()
