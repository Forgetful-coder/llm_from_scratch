from multi_head_attention.multi_head_attention import MultiHeadAttention
from layer_norm_gelu_activation.gelu_activation import Gelu
from layer_norm_gelu_activation.layer_norm import LayerNorm
from constants.constant import GPT_CONFIG_124M
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            Gelu(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):

    def __init__(self, cfg):
        
        self.attn = MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            num_heads=cfg['num_heads'],
            context_length=cfg['context_length'],
            droput=cfg['drop_rate']
)
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self,x):
        shortcut = x
        ## First part of the block
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x+shortcut

        #Second Part of x
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x+shortcut

        return x

