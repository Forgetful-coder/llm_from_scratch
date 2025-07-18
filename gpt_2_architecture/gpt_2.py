import torch
import torch.nn as nn
from layer_norm_gelu_activation.layer_norm import LayerNorm
from transformer_block.transformer import TransformerBlock
from constants.constant import GPT_CONFIG_124M


class GPT2(nn.Module):

    def __init__(self,cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'],cfg['emb_dim'])
        self.trans = nn.Sequential(
            *[TransformerBlock for _ in range(cfg['n_layers'])]
        )
        self.drop_rate = nn.Dropout(cfg['drop_rate'])
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'],cfg['vocab_size'], bias=False)
      


    def forward(self,in_seq):
        batch , seq = in_seq.shape
        token_emb = self.token_emb(in_seq)
        pos_emb = self.pos_emb(torch.arange(seq, device=in_seq.device))
        x = token_emb + pos_emb
        x = self.drop_rate(x)
        x = self.trans(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

        
