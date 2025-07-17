import torch
import torch.nn as nn


class Gelu(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        
        act = 0.5 * x * (1+torch.tanh(
             torch.sqrt(torch.tensor(2/torch.pi)) * 
             (x * 0.044715 * torch.pow(x,3))
        ))

        return act

    