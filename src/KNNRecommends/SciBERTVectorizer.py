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
        """
        sciBERT-Model used to create vector representations of titles and abstracts.

        Parameters
        ----------
        tokenizer : BertTokenizerFast, optional
            Tokenizer used to tokenizer stuff for the sciBERT-Model. Can be found on huggingface.
            The default is AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased").
        model : BertModel, optional
            Pretrained sciBERT-Mode. Can be found on huggingface.
            The default is AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").

        Returns
        -------
        None.

        """
        self.tokenizer = tokenizer
        self.model = model
        
        # check if GPU can be used
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to(self.device) ## model to GPU
            
        else:
            print("\n"*10+"Warning, no GPU detected, processing will take time..."+"\n"*4, flush=True)
            self.device = "cpu"
            
    def vectorize(self, texts, max_length = 512):
        """
        Vectroize texts using the sciBERT-Model

        Parameters
        ----------
        texts : str
            String sequence.
        max_length : int, optional
            Maximal amount of tokens that will be used for predictions.
            The default is 512.

        Returns
        -------
        np.array
            768-dimensional vector.

        """
        # if texts is a list of texts, then iterative over elements
        if type(texts) == list:
            
            text_tokens_ids = list(map(lambda t: self.tokenizer.encode(t,  max_length = max_length, truncation=True), texts))
            input_ids = torch.tensor(text_tokens_ids, device=self.device)
        
        # otherwise just transform one text
        elif type(texts) == str:
            input_ids = torch.tensor(self.tokenizer.encode(texts, max_length = max_length, truncation=True),\
                                     requires_grad=False, device=self.device).unsqueeze(0)  # Batch size 1
        
        # forard pass thorugh the BERT-model
        outputs = self.model(input_ids)
        last_hidden_states = outputs[0]

        # average each token-vector to a representation of all tokens and send it to the CPU (numpy)
        return last_hidden_states.cpu().detach().numpy().mean(1)
    