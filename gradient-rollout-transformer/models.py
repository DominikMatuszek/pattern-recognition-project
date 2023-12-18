import torch
from torch import Tensor
import torch.nn as nn
import timm


class trans_student(nn.Module):
    def __init__(self, **kwargs):
        super(trans_student, self).__init__()
        self.out_height: int = kwargs["out_height"]
        self.out_width: int = kwargs["out_width"]

        self.base = timm.create_model(kwargs["base_model_name"], pretrained=True)

        self.linear = nn.Linear(
            kwargs["base_model_out"], self.out_height * self.out_width
        )
        self.activation = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            x = self.base(x)
        x = self.linear(x)
        x = x.reshape(x.shape[0], self.out_width, self.out_height)
        x = self.activation(x)
        return x


def load_teacher(model_name, **kwargs):
    return timm.create_model(model_name, pretrained=True)