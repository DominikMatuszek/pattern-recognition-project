import torch 
import timm 

from validate_model import Validator
from optimal_sampler import OptimalSampler

def main():
    validator = Validator("optimal_dropout.txt")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    for dropout in range(0, 10):
        dp = dropout / 10

        model = OptimalSampler("vit_base_patch16_224", dropout_rate=dp, device=device)
        validator.log_accuracy_for_model(model, device, f"VIT with optimal sampling dropout {dp}")

if __name__ == "__main__":
    main()