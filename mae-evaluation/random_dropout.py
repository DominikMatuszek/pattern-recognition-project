import torch 
import timm 

from validate_model import Validator

def main():
    validator = Validator("random_dropout.txt")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    for dropout in range(1, 10):
        dp = dropout / 10

        vit = timm.models.create_model("vit_base_patch16_224", pretrained=True, patch_drop_rate=dp)
        validator.log_accuracy_for_model(vit, device, f"VIT with random dropout {dp}")

if __name__ == "__main__":
    main()