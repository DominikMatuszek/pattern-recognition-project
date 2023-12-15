import torch 
import torchvision

import timm 

class InterpolatingMAE(torch.nn.Module):
    def __init__(self, model, mae_out_dim, out_dim):
        super(InterpolatingMAE, self).__init__()
        self.model = model 
        self.projection = torch.nn.Linear(mae_out_dim, out_dim * out_dim)
        self.out_dim = out_dim
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)
        x = self.projection(x)
        x = x.reshape(x.shape[0], self.out_dim, self.out_dim)
        x = self.activation(x)
        return x
    
def main():
    base_mae = timm.create_model('vit_base_patch16_224.mae', pretrained=True)

    mae = InterpolatingMAE(base_mae, 768, 14)
    
    img = torch.randn(16, 3, 224, 224)

    out = mae(img)

    print(out.shape)

if __name__ == "__main__":
    main()

        