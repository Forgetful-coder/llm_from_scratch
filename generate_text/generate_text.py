import torch
import torch.nn as nn
from utils.utils import tokenize


class GenerateText:
    
    def generate_text(self,model,idx,context_size, max_new_tokens,temp=0.5,use_sampling=True,top_k=5):

        tokenizer = tokenize()

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -context_size:]

            with torch.no_grad():
                logits = model(idx_cond)
                if top_k > 0:
                    top_k_samp,top_k_pos = torch.topk(logits, top_k)
                    min_val = top_k_samp[:, -1]
                    logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)
                if temp > 0.0:
                    logits = logits/temp

                

                probas = torch.softmax(logits, dim=-1) #(batch_size, context_len)

                if use_sampling:
                    idx_next = torch.multinomial(probas,num_samples=1) #(batch_size,1)
                else:
                    idx_next = torch.argmax(probas,dim=-1,keepdim=True)

                idx = torch.cat((idx,idx_next), dim=-1)
        
        text = tokenizer.decode(idx.squeeze(0).tolist())

        return text

        