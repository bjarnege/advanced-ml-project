# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 21:13:48 2021

@author: Bjarne Gerdes
"""
import numpy as np
import pandas as pd
from scipy import spatial
from sklearn.neighbors import NearestNeighbors


class FindNeighbors:
    
    def __init__(self, author_graph, X_raw, X_raw_filtered, vectorizer_pipeline_words, vectorizer_pipeline_images,\
                 X_titles, X_abstracts, X_img, k):
        # Load transformation pipelines
        self.vectorizer_pipeline_words = vectorizer_pipeline_words
        self.vectorizer_pipeline_images = vectorizer_pipeline_images
        
        # Load Metadatafiles and the author graph
        self.X_raw = X_raw
        self.X_raw_filtered = X_raw_filtered
        self.author_graph = author_graph
    
        # Load precalculated vectors        
        self.X_titles = X_titles
        self.X_abstracts = X_abstracts
        self.X_img = X_img

        # k used for KNN
        self.k = k
        
    def cos(self, x_1, x_2):
        return spatial.distance.cosine(x_1, x_2)
        
    def fit(self, coauthor_paper_ids=None):
        
        # Fit the KNN model based on title BERT-vectors
        self.neigh_titles = NearestNeighbors(n_neighbors=self.k, metric=self.cos)
        
 
        # Fit the KNN model based on abstract BERT-vectors
        self.neigh_abstracts = NearestNeighbors(n_neighbors=self.k, metric=self.cos)
    
        # Fit the KNN model based on img vectors
        self.neigh_images = NearestNeighbors(n_neighbors=self.k, metric=self.cos)
    
        if coauthor_paper_ids is not None:
            self.neigh_titles.fit(self.X_titles.loc[coauthor_paper_ids])
            self.neigh_abstracts.fit(self.X_abstracts.loc[coauthor_paper_ids])
            
            img_indexes = set(self.X_img.index).intersection(coauthor_paper_ids)

            self.neigh_images.fit(self.X_img.loc[img_indexes])
        else:
            self.neigh_titles.fit(self.X_titles)
            self.neigh_abstracts.fit(self.X_abstracts)
            self.neigh_images.fit(self.X_img)            
            
            
    def get_neighbors_df(self, vector, knn):
        cosine, neighbors_indexes = knn.kneighbors(vector)
        neighbors = pd.DataFrame(self.X_raw.iloc[neighbors_indexes[0]])
        neighbors["cos sim"] = 1 - cosine[0]
        # filter all neighbors that are the same as the submitted paper
        neighbors = neighbors[neighbors["cos sim"] < 1]
        # exclude top 10 neighbors        
        return neighbors.head(10)

    
    def kneighbors(self, title_text, abstract_text, url_pdf, paper_id,
                   results, pipeline=["coauthors", "coauthor_filter",
                                      "titles", "abstracts", "images"]):
        
        if "coauthors" in pipeline or "coauthor_filter" in pipeline:
            ids = self.X_raw.loc[paper_id]["author_id"]
            all_co_authors = set(np.array(list(self.author_graph.edges(ids)))[:,1])
            filters = self.X_raw_filtered["author_id"].apply(lambda x: bool(set(all_co_authors) & set(x)))
            
            if "coauthors" in pipeline:
                if sum(filters) > 11:
                    results["coauthors"] = self.X_raw_filtered[filters].sample(11)
                else:
                    results["coauthors"] = self.X_raw_filtered[filters].to_dict()

            if "coauthor_filter" in pipeline:
                # Make sure enough co-author-matches will be found to have a sufficient
                # amount of papers for qualitative recommendations
                # uses 11 matches because 10 are co-author-matches and 1 can be the paper itself.
                if sum(filters) < 11:
                    results["comments"].append("Less than 10 Co-author-matches found in our dataset. "+
                                               "To ensure our quality standards, the co-author-filter will be disabled.")
                else:    
                    self.fit(coauthor_paper_ids=list(self.X_raw_filtered[filters].index))
                
        if "titles" in pipeline:
            vector_title = self.vectorizer_pipeline_words.vectorize(title_text)
            results["title"] = self.get_neighbors_df(vector_title, self.neigh_titles).to_dict()
            
        if "abstracts" in pipeline:
            vector_abstracts = self.vectorizer_pipeline_words.vectorize(abstract_text)
            results["abstracts"] = self.get_neighbors_df(vector_abstracts, self.neigh_abstracts).to_dict()
            
        if "images" in pipeline:
            try:
                vector_images = self.vectorizer_pipeline_images.vectorize(url_pdf)
                results["images"] = self.get_neighbors_df([vector_images], self.neigh_images).to_dict()
            except:
                results["comments"].append("Unable to do image-based recommendations due to missing "+\
                                           "images in Co-Author papers or the paper itself. "+\
                                           "You should consider to disable the co-author-filter. "+\
                                           "Note: Even a paper with images can lead to this error, "+\
                                           "because the images contained in the paper can't be "+\
                                           "extracted")
                
        # refit to ensure to forget about coauthors after calculations
        if "coauthor_filter" in pipeline:
            self.fit()
            
        return results
