import torch


class MishCuda(torch.nn.Module):  # https://github.com/digantamisra98/Mish
    def forward(self, x):
        return x * torch.nn.functional.softplus(x).tanh()
