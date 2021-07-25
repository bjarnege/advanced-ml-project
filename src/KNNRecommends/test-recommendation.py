# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 18:50:03 2021

@author: Bjarne Gerdes
"""

import pickle
import pandas as pd
import numpy as np
from recommmender import FindNeighbors

#%% Do modelling stuff
from SciBERTVectorizer import VectorizeData as textvectorizer
from VAEVectorizer import VectorizeData as imagevectorizer


author_graph = pickle.load(open("../../resource/metadata/graph.pkl", "rb"))

X_raw = pd.read_pickle("../../resource/metadata/metadata_df.pkl")
X_raw_filtered = pd.read_pickle("../../resource/metadata/metadata_df_filtered.pkl")

X_title_vectors = pd.read_pickle("../../resource/transformed_data/title_vectors.pkl")
X_abstract_vectors = pd.read_pickle("../../resource/transformed_data/abstract_vectors.pkl")

data_vae = np.load("../../resource/vae_data/encoded_images.pkl", allow_pickle=True)
X_image_vectors = pd.DataFrame(data=data_vae["encoded_pictures"], index=data_vae["paper_ids"]).groupby(data_vae["paper_ids"]).mean()
del(data_vae)

vectorizer_pipeline_words = textvectorizer()
vectorizer_pipeline_images = imagevectorizer()


fn = FindNeighbors(author_graph, X_raw, X_raw_filtered, vectorizer_pipeline_words, vectorizer_pipeline_images,\
                 X_title_vectors, X_abstract_vectors, X_image_vectors, 11)

fn.fit()
    
url_pdf = "https://arxiv.org/pdf/1306.0269.pdf"
title = "Experimental measurements of stress redistribution in flowing emulsions"
abstract = """We study how local rearrangements alter droplet stresses within flowing dense quasi-twodimensional emulsions at area fractions, Using microscopy, we measure droplet positions
while simultaneously using their deformed shape to measure droplet stresses. We find that rearrangements alter nearby stresses in a quadrupolar pattern: stresses on neighboring droplets tend
to either decrease or increase depending on location. The stress redistribution is more anisotropic
with increasing. The spatial character of the stress redistribution influences where subsequent
rearrangements occur. Our results provide direct quantitative support for rheological theories of
dense amorphous materials that connect local rearrangements to changes in nearby stress"""
paper_id = "1306.0269"

test = fn.kneighbors(title, abstract, url_pdf, paper_id)

