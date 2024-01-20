"""
Below code comes from the jacobgil/vit-explain repository with our modifications
Licensed under the MIT License
"""

""" 
MIT License

Copyright (c) 2020 Jacob Gildenblat

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import numpy as np
from config import IMAGENET_VAL_DIR
import torch.nn as nn
import resnet
from torchvision.transforms.functional import resize


def create_student():
    model_student = resnet.ResNet(input_shape=[1, 3, 90, 90], depth=26, base_channels=6)  ## ~ 160k parameters

    return model_student

def rollout(attentions, discard_ratio, head_fusion):
    # attentions = [attention.unsqueeze(0) for attention in attentions] # type: list[torch.Tensor]
    device = attentions[0].device

    result = torch.eye(attentions[0].size(-1), device=device)
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)  # type: torch.Tensor
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]  # type: torch.Tensor
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]  # type: torch.Tensor
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token

            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)

            # Flatten the last 2 dimensions

            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            cls_token_values = flat[:, 0]

            for i in range(flat.size(0)):
                flat[i, indices[i]] = 0
                flat[i, 0] = cls_token_values[i]

            I = torch.eye(attention_heads_fused.size(-1), device=device)
            a = (attention_heads_fused + 1.0 * I) / 2
            summed = a.sum(dim=-1).squeeze(1)
            a = torch.stack([a[i] / summed[i] for i in range(a.size(0))], dim=0)
            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches
    mask = result[:, 0, 1:]

    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1) ** 0.5)
    batch = mask.size(0)
    mask = mask.reshape(batch, width, width).cpu().numpy()
    maxed = np.max(mask, axis=(1, 2))
    maxed.shape = (batch, 1, 1)

    mask = mask / maxed
    return mask


class VITAttentionRollout:
    def __init__(
        self,
        model,
        attention_layer_name="attn_drop",
        head_fusion="mean",
        discard_ratio=0.9,
        device="cpu",
    ):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.device = device
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.to(self.device))

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)

        att = torch.stack(self.attentions)

        results = rollout(att, self.discard_ratio, self.head_fusion)
        return torch.tensor(results)


def main():
    import timm
    from imagenet_validation import ImageNet
    from matplotlib import pyplot as plt

    model = timm.models.create_model(
        "eva02_tiny_patch14_224.mim_in22k", pretrained=True
    )

    for block in model.blocks:
        block.attn.fused_attn = False

    # rollout = VITAttentionRollout(model, discard_ratio=0.8)

    ds = ImageNet(IMAGENET_VAL_DIR, size=224)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=2, shuffle=False, pin_memory=False
    )
    student = create_student()
    weights = torch.load("./deit_tiny_mean.pth")
    student.load_state_dict(weights)
    rollout = student

    for imgs, labels in loader:
        imgs = resize(imgs,size=(90,90),antialias=True)
        print(imgs.shape)
        mask = rollout(imgs)
        figs, axes = plt.subplots(2, 2)
        axes[0][0].imshow(imgs[0].permute(1, 2, 0))
        axes[0][1].imshow(mask[0].detach().numpy().reshape((14,14)))
        axes[1][0].imshow(imgs[1].permute(1, 2, 0))
        axes[1][1].imshow(mask[1].detach().numpy().reshape((14,14)))
        plt.show()

    print(mask.shape)


if __name__ == "__main__":
    main()
