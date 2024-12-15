import torch
import torch.nn as nn
from .configuration import ViT_Config
from .transformer import TransformerBlock
from transformers import ViTForImageClassification


class ViT(nn.Module):
    def __init__(self, config):
        super(ViT, self).__init__()
        assert config.image_size % config.patch_size == 0

        self.embed_dim = config.embed_dim

        #derive:
        self.num_channels = 3
        self.num_patches = (config.image_size//config.patch_size)**2
        
        #embed for cls this does not come from embedding fn:
        self.register_buffer("cls_token", torch.ones(1, 1, config.embed_dim))

        #Store position embedding as a parameter:
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches+1, self.embed_dim))

        #For patch embedding fn:
        self.projection = nn.Conv2d(
            self.num_channels, 
            config.embed_dim, 
            kernel_size=config.patch_size, 
            stride=config.patch_size
        )
        self.dropout = nn.Dropout(0.1, inplace=True) #just after embeddings - for training

        self.transformer = nn.Sequential(*[TransformerBlock(
                        self.num_patches, 
                        config,
        ) for _ in range(config.layers)]
        )

        self.ln_f = nn.LayerNorm(config.embed_dim) #layer norm after every block passes on

        self.head = nn.Linear(config.embed_dim, 1000) #classification head


    def patch_embeddings(self, x):
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1) #batch_size, 1, embed_dim
        
        # import code;code.interact(local=locals())
        x = self.projection(x) #batch_size, embed_dim, patch_num, patch_num
        # import code;code.interact(local=locals())
        x = x.view(x.shape[0], self.embed_dim, -1).permute(0, 2, 1) #batch_size, no_of_patches, embed_dim
        x = torch.cat((cls_tokens, x), dim=1)

        return x

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in ('google/vit-base-patch16-224')
        
        config_args = {
            'google/vit-base-patch16-224' : dict(embed_dim=768,  ff_dim=768*4,  num_heads=12, layers=12),
        }[model_type]
        patch_size = 16
        config_args['image_size'] =224
        config_args['patch_size'] = 16
        # import code;code.interact(local=locals())

        #vanilla vit dont have classifier head instead has pooler;
        model_hf = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224") #finetuned on imagenet
        sd_hf = model_hf.state_dict()
        #The original vit was trained on 21k imagenet, whose head is not released, only body;

        config = ViT_Config( **config_args)
        model=cls(config)
        sd = model.state_dict()

        assert len(sd) == len(sd_hf), "mismatch state dict, maybe you forgot to consider something"

        up = lambda i: {
            f'vit.encoder.layer.{i}.attention.attention.query.weight':      f'transformer.{i}.query.weight',
            f'vit.encoder.layer.{i}.attention.attention.query.bias':        f'transformer.{i}.query.bias',
            f'vit.encoder.layer.{i}.attention.attention.key.weight':        f'transformer.{i}.key.weight',
            f'vit.encoder.layer.{i}.attention.attention.key.bias':          f'transformer.{i}.key.bias',
            f'vit.encoder.layer.{i}.attention.attention.value.weight':      f'transformer.{i}.value.weight',
            f'vit.encoder.layer.{i}.attention.attention.value.bias':        f'transformer.{i}.value.bias',
            f'vit.encoder.layer.{i}.attention.output.dense.weight':         f'transformer.{i}.c_proj.weight',
            f'vit.encoder.layer.{i}.attention.output.dense.bias':           f'transformer.{i}.c_proj.bias',
            f'vit.encoder.layer.{i}.intermediate.dense.weight':             f'transformer.{i}.mlpf.c_fc.weight',
            f'vit.encoder.layer.{i}.intermediate.dense.bias':               f'transformer.{i}.mlpf.c_fc.bias',
            f'vit.encoder.layer.{i}.output.dense.weight':                   f'transformer.{i}.mlpf.c_proj.weight',
            f'vit.encoder.layer.{i}.output.dense.bias':                     f'transformer.{i}.mlpf.c_proj.bias',
            f'vit.encoder.layer.{i}.layernorm_before.weight':               f'transformer.{i}.ln1.weight',
            f'vit.encoder.layer.{i}.layernorm_before.bias':                 f'transformer.{i}.ln1.bias',
            f'vit.encoder.layer.{i}.layernorm_after.weight':                f'transformer.{i}.ln2.weight',
            f'vit.encoder.layer.{i}.layernorm_after.bias':                  f'transformer.{i}.ln2.bias',
        }


        

        mapping = {
            'vit.embeddings.cls_token': 'cls_token',
            'vit.embeddings.position_embeddings': 'position_embeddings',
            'vit.embeddings.patch_embeddings.projection.weight': 'projection.weight',
            'vit.embeddings.patch_embeddings.projection.bias': 'projection.bias',
            'vit.layernorm.weight': 'ln_f.weight',
            'vit.layernorm.bias': 'ln_f.bias',
            'classifier.weight': 'head.weight',
            'classifier.bias': 'head.bias'
        }

        for i in range(config_args['layers']): mapping.update(up(i))
        assert len(mapping.keys()) == len(sd_hf.keys()), "mismatch mapping between the models"

        from tqdm import tqdm
        print("Importing ViT")
        for k in tqdm(sd_hf):
            kn = mapping[k]
            assert sd_hf[k].shape == sd[kn].shape;
            with torch.no_grad():
                sd[kn].copy_(sd_hf[k])

        return model

    def forward(self, x):
        x = self.dropout(self.position_embeddings + self.patch_embeddings(x))
        print("After each transformer block:")
        print(x.shape)
        print("------------")
        x = self.ln_f(self.transformer(x))
        print("After taking out cls token through head:")
        return self.head(x[:,0,:])
