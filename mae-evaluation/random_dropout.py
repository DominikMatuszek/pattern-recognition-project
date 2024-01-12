import torch 
import timm 

from validate_model import Validator
from load_mae import load_mae

def main():
    validator = Validator("random_dropout.txt")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # print timm lsit
    #print(timm.list_models(pretrained=True))

    for dropout in range(0, 10):
        dp = dropout / 10

        #vit = timm.models.create_model("vit_large_patch14_clip_224.openai_ft_in12k_in1k", pretrained=True, pos_drop_rate=dp)
        vit = timm.models.create_model("vit_large_patch14_clip_224.openai_ft_in12k_in1k", pretrained=True, patch_drop_rate=dp)
        #vit = timm.models.create_model("deit3_tiny_patch16_224", pretrained=True, patch_drop_rate=dp)
        #vit = timm.models.create_model("deit_tiny_patch16_224.fb_in1k", pretrained=True, patch_drop_rate=dp)
        #vit = load_mae(patch_drop_rate=dp)
        validator.log_accuracy_for_model(vit, device, f"VIT with random dropout {dp}")

if __name__ == "__main__":
    main()