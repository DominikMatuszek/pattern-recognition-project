import torch 
import timm 

from validate_model import Validator
from optimal_sampler import OptimalSampler


def main():
    validator = Validator("optimal_dropout.txt", size=224)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    for dropout in range(0, 1):
        dp = dropout / 10
        
        model = OptimalSampler(
            "vit_large_patch14_clip_224.openai_ft_in12k_in1k",
            dropout_rate=dp,
            device=device,
            query_model_override="timm/eva02_tiny_patch14_224.mim_in22k",
        )

        validator.log_accuracy_for_model(model, device, f"VIT with optimal sampling dropout {dp}")

if __name__ == "__main__":
    main()