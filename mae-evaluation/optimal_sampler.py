import torch 
import timm 

from vit_rollout import VITAttentionRollout

class SaliencyDropout(torch.nn.Module):
    def __init__(self, dropout_rate, device='cpu'):
        super(SaliencyDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.device = device

    def forward(self, x):
        masked = []

        b = x.shape[0]

        for i in range(b):
            mask = self.mask[i]
            current = x[i]

            # Do not mess with the CLS token
            cls_token = current[0]
            current = current[1:]

            selected = torch.index_select(current, 0, mask)
            selected = torch.cat([cls_token.unsqueeze(0), selected], dim=0)

            masked.append(selected)

        x = torch.stack(masked, dim=0)

        return x 
    
    def set_mask(self, mask : torch.Tensor):
        self.mask = mask

        mask_length = mask.shape[1]
        after_dropout = int(mask_length * (1 - self.dropout_rate))
        _, indices = mask.topk(after_dropout, -1, largest=True, sorted=True)

        indices = indices.to(self.device)

        self.mask = indices 

class OptimalSampler(torch.nn.Module):
    def __init__(self, modelname, dropout_rate=0.5, device='cpu'):
        super(OptimalSampler, self).__init__()
        query_model = timm.models.create_model(modelname, pretrained=True).to(device)
        model = timm.models.create_model(modelname, pretrained=True).to(device)

        self.query_model = query_model

        for block in query_model.blocks:
            block.attn.fused_attn = False

        self.rollout = VITAttentionRollout(query_model)


        self.dropout = SaliencyDropout(dropout_rate, device=device)
        model.patch_drop = self.dropout 
        
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

    sampler = OptimalSampler("vit_base_patch16_224", dropout_rate=0.0)

    image_path = "../imagenet/ILSVRC2012_val_00050000.JPEG"
    image = Image.open(image_path)

    image = image.resize((224, 224))
    image = torchvision.transforms.ToTensor()(image)

    image = image.unsqueeze(0)

    pred = sampler(image)

    print(pred.argmax(dim=1))

if __name__ == "__main__":
    main()