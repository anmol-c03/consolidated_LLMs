import torch
import torch.nn as nn
from torch.nn import functional as F
from .mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, max_length, config, dropout=0.1):

        super(TransformerBlock, self).__init__()
        assert config.embed_dim % config.num_heads == 0, "embed_dim must be divisble by num_heads"

        self.max_length = max_length
        self.embed_dim = config.embed_dim
        self.ff_dim = config.ff_dim
        self.num_heads = config.num_heads
        self.dp = dropout

        #derv:
        self.head_size = self.embed_dim // self.num_heads

        #attention blocks
        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)

        #feedforward blocks
        self.mlpf = MLP(config)
        
        #after attn and ff blocks
        self.dropout = nn.Dropout(self.dp, inplace=True)

        #depends
        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.ln2 = nn.LayerNorm(self.embed_dim)

    def attn(self, x):

        batch_size, seq_length = x.shape[:2]


        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        # import code;code.interact(local=locals())

        Q = Q.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        # import code;code.interact(local=locals())

        att = torch.einsum('bhqd,bhkd->bhqk', [Q, K])/(self.head_size ** 0.5) #scaled dot produt attention
        att = F.softmax(att, dim=-1)

        out = torch.einsum('bhal,bhlv->bhav', [att, V]).permute(0,2,1,3).contiguous()
        out = out.view(batch_size, -1, self.num_heads * self.head_size)
        out = self.dropout(self.c_proj(out)) #projection after attending to tokens
        return out
    

    def forward(self, x):

        x = x + self.attn(self.ln1(x))
        x = x + self.mlpf(self.ln2(x))
        return x
