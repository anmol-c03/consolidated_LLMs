## This is the custom implementation of LORA(Lower Rank Adaptation)
## For futher details, one can consult this link https://arxiv.org/pdf/2106.09685

##main gist of this paper is :
'''
When adapting to a specific task, Inspired by this, 
it is hypothesized that the updates to the weights also have a low “intrinsic rank” .
W0 represents the weights of the pretrained model.
∆W is weights of added LoRA layers
such that 
h = W0 + ∆W  = W0 + BA

where  ∆W= BA
A is  a random Gaussian initialization 
B is initially initialied to zero
'''

#in this implementation i have ignored the aplha value for simplicity
import torch
import torch.nn as nn


from transformers import AutoModelForCausalLM,AutoTokenizer,TrainingArguments,trainer,BitsAndBytesConfig

# #just ignore this section
bnb_4bit_compute_dtype='float16'
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

# #just ignore this section
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)
# #upto here 

model_name="NousResearch/llama-2-7b-chat-hf"

model=AutoModelForCausalLM.from_pretrained(model_name,
                           quantization_config=bnb_config)

tokenizer=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side='right'
# model=nn.ModuleDict(dict(
#                     ln1=nn.Linear(12,32),
#                     re=nn.ReLU(),
#                     ln2=nn.Linear(32,16)))

# sd=model.state_dict()
# for k,v in sd.items():
#     print(k,v.shape)

class custom_LORA(nn.Module):
    def __init__(self,original_layer,rank=4):
        super(custom_LORA,self).__init__()
        self.original_layer=original_layer
        self.rank=rank
        self.A=nn.Parameter(torch.randn(self.rank,self.original_layer.in_features))
        self.B=nn.Parameter(torch.zeros(self.original_layer.out_features,self.rank))

    def forward(self,x):
        return self.original_layer(x)+self.x @ self.A.T @ self.B.T
    
# here we are replacing just for example the linear layer with lora 
for name,module in model.named_modules():
    print(module)
    if isinstance(module,nn.Linear):
        setattr(model,name,custom_LORA(module))
    
for params in model.parameters():
    params.requires_grad=False

for name,module in model.named_modules():
    if isinstance(module,custom_LORA):
        module.A.requires_grad=True
        module.B.requires_grad=True
# sd=model.state_dict()
# for k,v in sd.items():
#     print(k,v.shape)

# after this we just have to follow FiT_lamma2 
