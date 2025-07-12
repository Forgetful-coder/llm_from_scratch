import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_in, d_out, num_heads,context_length, droput = 0.0, qkv_bias = False):

        assert(d_out%num_heads==0)
        """
        d_out must be divisible by num_heads for head_dim
        """
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_q = nn.Linear(d_in,d_out, bias = qkv_bias)
        self.W_k = nn.Linear(d_in,d_out, bias = qkv_bias)
        self.W_v = nn.Linear(d_in,d_out, bias = qkv_bias)
        self.out_proj = nn.Linear(d_out,d_out)
        self.dropout = nn.Dropout(droput)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length,context_length), diagonal=1)
        )

    def forward(self,x):
        b, num_token, d_in = x.shape

        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        queries = queries.view(b, num_token, self.num_heads,self.head_dim)
        keys = keys.view(b, num_token, self.num_heads,self.head_dim)
        values = values.view(b, num_token, self.num_heads,self.head_dim)

        #Group by num_heads
        queries = queries.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        #Calculate scores
        attn_scores = queries @ keys.transpose(2,3)

        mask_bool = self.mask[:num_token, :num_token]

        attn_scores = attn_scores.masked_fill_(mask_bool,-torch.inf)

        attn_weights = torch.softmax(attn_scores/self.head_dim**0.5, dim=-1)
        #Dropout to Prevent overfitting and better generalizing
        attn_weights = self.dropout(attn_weights)

        
        context_vec = (attn_weights @ values).transpose(1, 2) 
        context_vec = context_vec.contiguous().view(b,num_token,self.d_out)

        context_vec  = self.out_proj(context_vec) #optional

        return context_vec




