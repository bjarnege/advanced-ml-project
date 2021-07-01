# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 15:08:21 2021

@author: Bjarne Gerdes
"""

import torch
from transformers import AutoTokenizer, AutoModel
  

class VectorizeData:
    
    def __init__(self, tokenizer =  AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased"),\
                 model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")):
        
        self.tokenizer = tokenizer
        self.model = model
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to(self.device) ## model to GPU
            
        else:
            print("Warning, no GPU detected, processing will take time...", flush=True)
            self.device = "cpu"
            
    def embed_text(self, texts, max_length = 512):
        
        if type(texts) == list:
            
            text_tokens_ids = list(map(lambda t: self.tokenizer.encode(t,  max_length = max_length, truncation=True), texts))
            input_ids = torch.tensor(text_tokens_ids, device=self.device)
            
        elif type(texts) == str:
            input_ids = torch.tensor(self.tokenizer.encode(texts, max_length = max_length, truncation=True),\
                                     requires_grad=False, device=self.device).unsqueeze(0)  # Batch size 1
        
        outputs = self.model(input_ids)
        last_hidden_states = outputs[0]

    
        return last_hidden_states.cpu().detach().numpy().mean(1)
    