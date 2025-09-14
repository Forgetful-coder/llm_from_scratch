import torch
from torch.utils.data import Dataset


class SpamDataset(Dataset):
    def __init__(self,df,tokenizer,max_length,pad_token_id=50256):
        super().__init__()
        self.data = df
        self.max_length = max_length

        self.encoded_text = [
            tokenizer(text) for text in self.data["Text"]
        ]

        if self.max_length is None:
            self.max_length = self.longest_encoded_text()
        
        else:

            self.encoded_text = [
                encoded_text[:self.max_length] for \
                encoded_text in self.encoded_text
            ]
        
    def __getitem__(self,index):
            encoded = self.encoded_text[index]
            label = self.data.iloc[index]["Label"]

            return (
                 torch.tensor(encoded,dtype=torch.long),
                 torch.tensor(label, dtype=torch.long)
            )
    
    def __len__(self):
        return len(self.data)
    
    def longest_encoded_text(self):
        max_length = 0

        for encoded_text in self.encoded_text:
              if len(encoded_text) > max_length:
                   max_length = len(encoded_text)
        
        return max_length

        




