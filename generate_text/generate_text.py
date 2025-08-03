import torch
import torch.nn as nn
from utils.utils import tokenize


class GenerateText:
    
    def generate_text(self,model,idx,context_size, max_new_tokens):

        tokenizer = tokenize()

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -context_size:]

            with torch.no_grad():
                logits = model(idx_cond)

                last_vector = logits[:, -1, :]

                probas = torch.softmax(last_vector, dim=-1)

                idx_next = torch.argmax(probas,dim=-1,keepdim=True)

                idx = torch.cat((idx,idx_next), dim=-1)
        
        text = tokenizer.decode(idx.squeeze(0).tolist())

        return text

        