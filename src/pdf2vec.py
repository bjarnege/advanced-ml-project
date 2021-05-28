# -*- coding: utf-8 -*-
"""
Created on Tue May 25 18:32:49 2021

@author: Bjarne Gerdes
"""
import os
from glob import glob
import zipfile
from multiprocess  import Pool
import tqdm
from read_pdf import readPdf
from transformers import AutoTokenizer, AutoModel
import torch

class PdfEncoder:
    
    def __init__(self, output_filepath="./pdf_vectors.csv", input_folder_path="../data",\
                 unzip_tmp_path="../tmp"):
        self.output_filepath = output_filepath
        self.unzip_tmp_path = unzip_tmp_path
        self.files = [y.replace("\\","/") for x in os.walk(input_folder_path) for y in glob(os.path.join(x[0], '*.zip'))]
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

        if not os.path.exists(self.unzip_tmp_path):
            os.mkdir(self.unzip_tmp_path)
            
    def unzip(self, file):
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(self.unzip_tmp_path)
        
    def readOneZIP(self, file):
        # unzip into tmp folder
        self.unzip(file)
        # get all names of the unziped files
        self.pdf_files = [y.replace("\\","/") for x in os.walk(self.unzip_tmp_path) for y in glob(os.path.join(x[0], '*.pdf'))]
        # read all pdfs
        with Pool(4) as p:
            self.pdf_contents  = list(tqdm.tqdm(p.imap(readPdf, self.pdf_files), total=len(self.pdf_files)))
        
        for pdf_file in self.pdf_files:
            os.remove(pdf_file)
            
        os.remove(file)
        return self.pdf_contents
     
    def embed_text(self, content):
        input_ids = torch.tensor(self.tokenizer.encode(content,  max_length = 512)).unsqueeze(0)  # Batch size 1
        outputs = self.model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output
        
        return last_hidden_states

    def tokenizeList(self, contents):
        with Pool(4) as p:
            vectors  = list(tqdm.tqdm(p.imap(self.embed_text, contents), total=len(contents)))
    
        return vectors



test = PdfEncoder()
data = test.readOneZIP(test.files[0])
data_c = [d[1] for d in data]
vectors = test.tokenizeList(contents)
