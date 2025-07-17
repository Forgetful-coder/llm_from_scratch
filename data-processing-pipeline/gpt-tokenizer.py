import tiktoken
import torch
from torch.utils.data import DataLoader,Dataset


class GPTDatasetV1(Dataset):
    def __init__(self,txt,tokenizer,max_length,stride):
        self.input_ids = []
        self.target_ids = []


        tokenizer = tokenizer.encode(txt, allowed_special = {"<|endoftext|>"})

        for i in range(0, len(txt) - max_length, stride):
            input_ids = tokenizer[i:i+max_length]
            target_ids = tokenizer[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor[input_ids])
            self.target_ids.append(torch.tensor[target_ids])

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    tokenizer = tiktoken.get_encoding('gpt2')

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

def add_token_embeddings(vocab_size, output_dim):

    with open("data/the-verdict.txt", 'r', encoding = "utf-8" ) as f:
        txt = f.read()

    token_embeddings_layer = torch.nn.Embedding(vocab_size,output_dim)
    max_length = 4
    dataloader = create_dataloader_v1(
    txt, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
    )
    pos_embeddings_layer = torch.nn.Embedding(max_length,output_dim)
    pos_embeddings = pos_embeddings_layer(torch.arange(max_length))
    

    for inputs, targets in dataloader:

        input_embeddings = token_embeddings_layer(inputs)

        input_embeddings = input_embeddings + pos_embeddings


    
    
        