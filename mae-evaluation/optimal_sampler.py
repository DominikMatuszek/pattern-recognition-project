import torch 
import timm 

from vit_rollout import VITAttentionRollout

class SaliencyDropout(torch.nn.Module):
    def __init__(self, dropout_rate, device='cpu'):
        super(SaliencyDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.device = device

    def forward(self, x):
        mask = self.mask # (batch_size, num_tokens)
        # x shape: (batch_size, num_tokens, num_features)

        # We want to pick, for each element of batch, a subset of the token as specified by the mask
        x = torch.cat([x[i, mask[i], :].unsqueeze(0) for i in range(x.shape[0])], dim=0)

        return x 
    
    def set_mask(self, mask : torch.Tensor):
        mask_length = mask.shape[1]
        after_dropout = int(mask_length * (1 - self.dropout_rate))
        _, indices = mask.topk(after_dropout, -1, largest=True, sorted=False)

        indices = indices.to(self.device)

        # Add 1 to every index to account for the CLS token
        indices = indices + 1

        # Append the CLS token
        indices = torch.cat([torch.zeros(indices.shape[0], 1, dtype=torch.long, device=self.device), indices], dim=1)

        self.mask = indices 

class OptimalSampler(torch.nn.Module):
    def __init__(self, model, dropout_rate=0.5, device='cpu', query_model_override=None, drop_name='patch_drop'):
        super(OptimalSampler, self).__init__()

        # If model is a string, load the model from timm
        if isinstance(model, str):
            print("Initializing model from timm")
            modelname = model 
            query_model = timm.models.create_model(modelname, pretrained=True).to(device)
            model = timm.models.create_model(modelname, pretrained=True).to(device)

            if query_model_override is not None:
                print("Overriden query model")
                query_model = timm.models.create_model(query_model_override, pretrained=True).to(device)

        elif isinstance(model, torch.nn.Module):
            raise Exception("Model must be a string or function returning model; not a torch.nn.Module. Because we need two of them.")
        elif callable(model):
            # If model is a callable, assume it is a function that returns a model
            # Yes, I know, I know
            print("Initializing model from function")
            query_model = model().to(device)
            model = model().to(device)

            if query_model_override is not None:
                print("Overriden query model")
                query_model = query_model_override().to(device)

        self.query_model = query_model

        for block in query_model.blocks:
            block.attn.fused_attn = False

        self.rollout = VITAttentionRollout(query_model, discard_ratio=0.8, device=device)


        self.dropout = SaliencyDropout(dropout_rate, device=device)
        
        setattr(model, drop_name, self.dropout)
        
        #model.patch_drop = self.dropout 
        
        self.model = model


    def forward(self, x):
        saliency = self.rollout(x)
        saliency = torch.flatten(saliency, start_dim=1)
        self.dropout.set_mask(saliency)

        return self.model(x)
    
def main():
    from PIL import Image
    from matplotlib import pyplot as plt
    import torchvision

    sampler = OptimalSampler("vit_base_patch16_224", dropout_rate=0.5)

    image_path = "../imagenet/ILSVRC2012_val_00050000.JPEG"
    image = Image.open(image_path)

    image = image.resize((224, 224))
    image = torchvision.transforms.ToTensor()(image)

    image = image.unsqueeze(0)

    pred = sampler(image)

    print(pred.argmax(dim=1))

if __name__ == "__main__":
    main()