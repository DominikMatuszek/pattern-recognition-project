import torch 
from validate_model import Validator
from optimal_sampler import ProgrammableSampler
import resnet


def main():
    validator = Validator("optimal_dropout.txt", size=224)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    student = resnet.ResNet(input_shape=[1, 3, 90, 90], depth=26, base_channels=6) 
    weights = torch.load("./deit_tiny_mean.pth")
    student.load_state_dict(weights)


    for dropout in range(0, 10):
        dp = dropout / 10
        
        model = ProgrammableSampler(
            "vit_large_patch14_clip_224.openai_ft_in12k_in1k",
            dropout_rate=dp,
            device=device,
            query_model_override="timm/eva02_tiny_patch14_224.mim_in22k",
            saliency_fn=student
        )

        validator.log_accuracy_for_model(model, device, f"VIT with optimal sampling dropout {dp}")

if __name__ == "__main__":
    main()
