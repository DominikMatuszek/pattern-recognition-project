import torch 
import torchvision

import timm 

class StudentTransformer(torch.nn.Module):
    def __init__(self, base, base_out_dim, out_height, out_width):
        super(StudentTransformer, self).__init__()
        self.model = base 
        self.projection = torch.nn.Linear(base_out_dim, out_height * out_width)
        self.activation = torch.nn.Sigmoid()
        self.finetuning = False

        self.out_height = out_height
        self.out_width = out_width

    def forward(self, x):
        if not self.finetuning:
            with torch.no_grad():
                x = self.model(x)
        else:
            x = self.model(x)

        x = self.projection(x)
        x = x.reshape(x.shape[0], self.out_height, self.out_width)
        x = self.activation(x)
        return x
    
def main():
    base_mae = timm.create_model('vit_base_patch16_224.mae', pretrained=True)

    mae = StudentTransformer(base_mae, 768, 14, 14)
    
    img = torch.randn(16, 3, 224, 224)

    out = mae(img)

    print(out.shape)

if __name__ == "__main__":
    main()

        