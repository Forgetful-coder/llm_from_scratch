import os
import urllib.request
import torch
import torch.nn as nn
from constants.constant import GPT_CONFIG_124M, FILE_PATH, URL
from gpt_2_architecture.gpt_2 import GPT2
from data_processing_pipeline.gpt_tokenizer import create_dataloader_v1
import time

class Pipeline:
    def __init__(self,train_ratio=0.90):
        self.train_ratio = train_ratio
        self.text_data = self.read_data()
        split_idx = int(train_ratio * len(self.text_data))
        self.train_data = self.text_data[:split_idx]
        self.val_data = self.text_data[split_idx:]
    
    @staticmethod
    def read_data(self):
        if not os.path.exists(FILE_PATH):
            with urllib.request.urlopen(URL) as response:
                text_data = response.read().decode('utf-8')
            with open(FILE_PATH, "w", encoding="utf-8") as file:
                file.write(text_data)
        else:
            with open(FILE_PATH, "r", encoding="utf-8") as file:
                text_data = file.read()
    
    def create_data(self):
        train_loader = create_dataloader_v1(
        self.train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

        val_loader = create_dataloader_v1(
        self.val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    
        return train_loader, val_loader
    
    def calc_loss_batch(self,input_batch, target_batch, model, device):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)
        
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss


    def calc_loss_loader(self,data_loader, model, device, num_batches=None):
        total_loss = 0.
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            # Reduce the number of batches to match the total number of batches in the data loader
            # if num_batches exceeds the number of batches in the data loader
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = self.calc_loss_batch(input_batch, target_batch, model, device)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches
    
    def train_model_simple(self,model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context):
        # Initialize lists to track losses and tokens seen
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1   
    
        for epoch in range(num_epochs):
            model.train()

            for input_batch,target_batch in train_loader:
                optimizer.zero_grad()
                loss = self.calc_loss_batch(input_batch,target_batch,model,device)
                loss.backward()
                optimizer.step()
                tokens_seen+= input_batch.numel()
                global_step += 1
            
                if global_step % eval_freq == 0: 
                    train_loss, val_loss = self.evaluate_model(
                        model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            

        return train_losses, val_losses, track_tokens_seen
    
    def evaluate_model(self,model, train_loader, val_loader, device, eval_iter):
        model.eval()
        with torch.no_grad():
            train_loss = self.calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
            val_loss = self.calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()
        return train_loss, val_loss
    
    def run(self):
        train_loader, val_loader = self.create_data()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GPT2(GPT_CONFIG_124M)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

        num_epochs = 10
        train_losses, val_losses, tokens_seen = self.train_model_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs, eval_freq=5, eval_iter=5,
            start_context="Every effort moves you"
        )


if __name__ == "__main__":
    pipeline = Pipeline()
    start = time.time()
    pipeline.run()
    end = time.time()
    print(f"Total time taken to run is: {(end - start)/60:.2f} minutes")


    
    
    

                                       
    