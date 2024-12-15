import torch
import torch.nn as nn
import math

#torch gelu is different
class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc    = nn.Linear(config.embed_dim, config.ff_dim)
        self.newgelu    = NewGELU()
        self.c_proj  = nn.Linear(config.ff_dim, config.embed_dim)
    def forward(self,x):
        x=self.c_fc(x)
        x=self.newgelu(x)
        x=self.c_proj(x)
        # self.out=self.dropout(x)
        return x
    