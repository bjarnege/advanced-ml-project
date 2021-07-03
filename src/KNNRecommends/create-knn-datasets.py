# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 15:09:04 2021

@author: Bjarne Gerdes
"""
import os
import Vectorizer
import pandas as pd
from tqdm import tqdm
import numpy as np

#%% Read metadata
metadata_path = "../../resource/metadata/"
df_metadata = pd.read_pickle(metadata_path+"/metadata_df_filtered.pkl")

vectorizer = Vectorizer.VectorizeData()

#%% Vectorize abstract:
print("Vecotrizing abstracts ... \n", flush=True)
vectors_abstract = [vectorizer.embed_text(abstract) for abstract in tqdm(list(df_metadata["abstract"].values))]
    
#%% Vectorize titles:
print("Vectorizing titles ... \n", flush=True)
vectors_title = [vectorizer.embed_text(title) for title in tqdm(list(df_metadata["title"].values))]

#%% Convert and store vectors
df_vectors_abstract = pd.DataFrame(np.array(vectors_abstract).squeeze())
df_vectors_abstract.index = df_metadata.index

df_vectors_title = pd.DataFrame(np.array(vectors_title).squeeze())
df_vectors_title.index = df_metadata.index


output_path =  "../../resource/transformed_data"

if not os.path.exists(output_path):
    os.mkdir(output_path)
    
df_vectors_abstract.to_pickle(output_path+"/abstract_vectors.pkl")
df_vectors_title.to_pickle(output_path+"/title_vectors.pkl")
