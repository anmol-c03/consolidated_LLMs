# it is python implementation of rotational positional embedding

'''why RoPE?
because it decays with the relative distance increased,
which is desired for natural language encoding

Llama uses RoPE instead of absolute positional embedding'''

import torch
import numpy as np

from dataclasses import dataclass

@dataclass
class RoPE_config:
    batch_size=4
    block_size=2
    num_heads=2
    d_model=8

class Rotary_PE:
    def __init__(self,rope):
        self.batch_size=rope.batch_size
        self.block_size=rope.block_size
        self.num_heads=rope.num_heads
        self.d_model=rope.d_model
        self.pos_emb=torch.zeros(self.block_size,self.d_model)
        assert self.d_model%2==0

    def get_PE(self):
        positions=torch.arange(self.block_size)[:,np.newaxis]
        theta=10000**(-2*(torch.arange(1,self.d_model//2+1)-1)/self.d_model)
        self.pos_emb[:,0::2]=torch.cos(positions*theta)
        self.pos_emb[:,1::2]=torch.sin(positions*theta)
        self.pos_emb=self.pos_emb[np.newaxis,:,:]
    
    #qw and kw are the token embeddings 
    def __call__(self,x):
        cosine_term=torch.repeat_interleave(self.pos_emb[...,None,0::2],repeats=2,axis=-1)
        sine_term=torch.repeat_interleave(self.pos_emb[...,None,1::2],repeats=2,axis=-1)
        x_dash=torch.stack([-x[...,1::2],x[...,0::2]],axis=-1)
        x_dash=x_dash.view(x.shape)
        return x*cosine_term+x_dash*sine_term

rope=RoPE_config()
#let below represent token_embedding
# in actual paper rotational matrix is multiplied with token_emb but
#RoPE is a method to incorporate positional information into the self-attention mechanism by applying rotations to the query and key vectors based on their positionse
query=torch.randn(rope.batch_size,rope.block_size,rope.num_heads,rope.d_model)
key=torch.randn(rope.batch_size,rope.block_size,rope.num_heads,rope.d_model)

#intializing the Rotary_PE
model=Rotary_PE(rope)
model.get_PE()
query=model(query)
key=model(key)
attention_mat=torch.einsum('bjhd,bkhd->bhjk',query,key)

print(attention_mat.shape)
