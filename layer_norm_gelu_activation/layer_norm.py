import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    
    def __init__(self,emb_dim):
        super().__init__()
        self.eps  = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self,x):
        mean = torch.mean(x, dim=-1 ,keepdim=True)
        var = torch.var(x,dim=-1, keepdim=True, unbiased=False) ## To avoid applying Bessels correction(n-1)

        norm = (x-mean)/(torch.sqrt(var))

        return self.scale * norm + self.shift
        
