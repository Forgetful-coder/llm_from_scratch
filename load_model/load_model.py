from pre_trained.gpt2 import get_model
from constants.constant import GPT_CONFIG_124M
import torch
from gpt_2_architecture.gpt_2 import GPT2
import numpy as np



class LoadModel:

    def __init__(self):
        self.config = GPT_CONFIG_124M
        self.settings, self.params = get_model(model_size="124M", model_dir="models")
        self.new_config = self.config.copy()
        self.new_config = self.new_config.update({"qkv_bias":True})
        self.gpt = GPT2(self.new_config).eval()
    
    @staticmethod
    def assign(left,right):
        if left.shape!=right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(torch.tensor(right))
    
    def load_weights(self):
        self.gpt.pos_emb.weight = self.assign(self.gpt.pos_emb.weight, self.params['wpe'])
        self.gpt.token_emb.weight = self.assign(self.gpt.token_emb.weight, self.params['wte'])

        for b in range(len(self.params['blocks'])):
            q_w, k_w, v_w = np.split(
            (self.params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            
            self.gpt.trf_blocks[b].att.W_query.weight = self.assign(
                self.gpt.trf_blocks[b].att.W_query.weight, q_w.T)
            self.gpt.trf_blocks[b].att.W_key.weight = self.assign(
                self.gpt.trf_blocks[b].att.W_key.weight, k_w.T)
            self.gpt.trf_blocks[b].att.W_value.weight = self.assign(
                self.gpt.trf_blocks[b].att.W_value.weight, v_w.T)

            q_b, k_b, v_b = np.split(
                (self.params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            self.gpt.trf_blocks[b].att.W_query.bias = self.assign(
                self.gpt.trf_blocks[b].att.W_query.bias, q_b)
            self.gpt.trf_blocks[b].att.W_key.bias = self.assign(
                self.gpt.trf_blocks[b].att.W_key.bias, k_b)
            self.gpt.trf_blocks[b].att.W_value.bias = self.assign(
                self.gpt.trf_blocks[b].att.W_value.bias, v_b)

            self.gpt.trf_blocks[b].att.out_proj.weight = self.assign(
                self.gpt.trf_blocks[b].att.out_proj.weight, 
                self.params["blocks"][b]["attn"]["c_proj"]["w"].T)
            self.gpt.trf_blocks[b].att.out_proj.bias = self.assign(
                self.gpt.trf_blocks[b].att.out_proj.bias, 
                self.params["blocks"][b]["attn"]["c_proj"]["b"])

            self.gpt.trf_blocks[b].ff.layers[0].weight = self.assign(
                self.gpt.trf_blocks[b].ff.layers[0].weight, 
                self.params["blocks"][b]["mlp"]["c_fc"]["w"].T)
            self.gpt.trf_blocks[b].ff.layers[0].bias = self.assign(
                self.gpt.trf_blocks[b].ff.layers[0].bias, 
                self.params["blocks"][b]["mlp"]["c_fc"]["b"])
            self.gpt.trf_blocks[b].ff.layers[2].weight = self.assign(
                self.gpt.trf_blocks[b].ff.layers[2].weight, 
                self.params["blocks"][b]["mlp"]["c_proj"]["w"].T)
            
            self.gpt.trf_blocks[b].ff.layers[2].bias = self.assign(
                self.gpt.trf_blocks[b].ff.layers[2].bias, 
                self.params["blocks"][b]["mlp"]["c_proj"]["b"])

            self.gpt.trf_blocks[b].norm1.scale = self.assign(
                self.gpt.trf_blocks[b].norm1.scale, 
                self.params["blocks"][b]["ln_1"]["g"])
            self.gpt.trf_blocks[b].norm1.shift = self.assign(
                self.gpt.trf_blocks[b].norm1.shift, 
                self.params["blocks"][b]["ln_1"]["b"])
            self.gpt.trf_blocks[b].norm2.scale = self.assign(
                self.gpt.trf_blocks[b].norm2.scale, 
                self.params["blocks"][b]["ln_2"]["g"])
            self.gpt.trf_blocks[b].norm2.shift = self.assign(
                self.gpt.trf_blocks[b].norm2.shift, 
                self.params["blocks"][b]["ln_2"]["b"])

        self.gpt.final_norm.scale = self.assign(self.gpt.final_norm.scale, self.params["g"])
        self.gpt.final_norm.shift = self.assign(self.gpt.final_norm.shift, self.params["b"])
        self.gpt.out_head.weight = self.assign(self.gpt.out_head.weight, self.params["wte"])