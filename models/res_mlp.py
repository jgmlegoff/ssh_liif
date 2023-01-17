import torch.nn as nn

from models import register

class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.coord = coord


@register('res_mlp')
class RESMLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        hidden = 256
        self.fc0 = nn.Linear(lastv,hidden)
        lastv = hidden
        self.fc = nn.Linear(lastv, hidden)
        self.fcout = nn.Linear(lastv, out_dim)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)