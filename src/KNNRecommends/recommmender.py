# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 21:13:48 2021

@author: Bjarne Gerdes
"""

import pandas as pd
import numpy as np
from scipy import spatial
from sklearn.neighbors import NearestNeighbors


class FindNeighbors:
    
    def __init__(self, X_raw, word2vec, X=None):
        self.word2vec = word2vec
        self.X_raw = X_raw
        
        if type(X) != None:
            self.X = X
        
        
    def cos(self, x_1, x_2):
        return spatial.distance.cosine(x_1, x_2)
        
    def fit(self, k):
        if not hasattr(self, 'X'):
            self.X = np.array([self.word2vec.\
                               embed_text(text)[0]\
                               for text in self.X_raw])

        self.neigh = NearestNeighbors(n_neighbors=k, metric=self.cos)
        self.neigh.fit(self.X)
    
    def kneighbors(self, text):
        vector = self.word2vec.embed_text(text)
        cosine, neighbors_indexes = self.neigh.kneighbors(vector)
        neighbors = pd.DataFrame(self.X_raw.iloc[neighbors_indexes[0]])
        neighbors["cos sim"] = 1 - cosine[0]
        return neighbors
