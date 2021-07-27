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
        """
        This function takes all required pickle files and can be used to
        generate recommendations.

        Parameters
        ----------
        author_graph : networkx.Graph
            Undirected co-author-graph.
            
        X_raw : pandas.DataFrame
            DataFrame containing all metadata.
            
        X_raw_filtered : pandas.DataFrame
            DataFrame containing the metadata about the papers, that are used to 
            generate recommendations.
            
        vectorizer_pipeline_words : class
            Instance of the class VectorizeData from the KNNRecommends.SciBERTVectorizer-File.
            Is used to transform a given str into a BERT-Vector-representation.
            
        vectorizer_pipeline_images : class
            Instance of the class VectorizeData from the KNNRecommends.VAEVectorizer-File.
            Is used to process a given str, containing an URL, by downloading the 
            pdf-file which can be found at that URL and extracting all images in this file.
            After that, every image will used as the anput for an varitional auto-encoder 
            and represenated as a z-vector. 
            In the final step all z-vectors will be averaged to create a vector representation
            of every image in the file.
            
        X_titles : pandas.DataFrame
            sciBERT-Vector of every title in the X_raw_filtered DataFrame.
            
        X_abstracts : pandas.DataFrame
            sciBERT-Vector of every abstract in the X_raw_filtered DataFrame.
            
        X_img : pandas.DataFrame
            VAE-imagevector of every paper in the X_raw_filtered DataFrame,
            that contain extractable images.
            
        k : int
            k used for the KNN-algorithm.

        Returns
        -------
        None.

        """
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
        """
        Cosine distance between two vector representations (image-, abstract-, titlevector).

        Parameters
        ----------
        x_1 : pd.Series or np.array
            Vector representation of paper A.
        x_2 : pd.Series or np.array
            Vector representation of paper B.


        Returns
        -------
        float
            cos(paper A, paper B)
        """
        return spatial.distance.cosine(x_1, x_2)
        
    def fit(self, coauthor_paper_ids=None):
        """
        Fits all three KNN models.

        Parameters
        ----------
        coauthor_paper_ids : list, optional
            If this paper is a list, it will be used to filter the KNN-results based on the condition
            if a paper written by a co-author.
            The default is None.

        Returns
        -------
        None.

        """
        # Init the KNN model based on title BERT-vectors
        self.neigh_titles = NearestNeighbors(n_neighbors=self.k, metric=self.cos)
        
 
        # Init the KNN model based on abstract BERT-vectors
        self.neigh_abstracts = NearestNeighbors(n_neighbors=self.k, metric=self.cos)
    
        # Init the KNN model based on img vectors
        self.neigh_images = NearestNeighbors(n_neighbors=self.k, metric=self.cos)
        
        # Fit the models and decide of co-authors will be used as a filter
        # or not
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
        """
        Takes a vector representation and uses KNN
        to find top 10 neighbors and returns them 
        as a pandas.DataFrame

        Parameters
        ----------
        vector : pd.Series or np.array
            Vector representation of a paper.
        knn : class
            KNN-instance that will be used.
            For example if the vector is a sciBERT-title vector, the 
            KNN for titles should be used.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the top 10 matches.

        """
        cosine, neighbors_indexes = knn.kneighbors(vector)
        neighbors = pd.DataFrame(self.X_raw_filtered.iloc[neighbors_indexes[0]])
        neighbors["cos sim"] = 1 - cosine[0]
        # filter all neighbors that are the same as the submitted paper
        neighbors = neighbors[neighbors["cos sim"] < 1]
        # return top 10 neighbors        
        return neighbors.head(10)

 
    def kneighbors(self, title_text, abstract_text, url_pdf, paper_id,
                   results, pipeline=["coauthors", "coauthor_filter",
                                      "titles", "abstracts", "images"]):
        """
        Perform the commendation part in the backend and is used to
        process the requests of the API to create recommendations.

        Parameters
        ----------
        title_text : str
            title of the paper.
        abstract_text : str
            abstract of the paper.
        url_pdf : str
            url of the paper.
        paper_id : str
            id of the paper.
        results : dict
            dict in which the recommendations will be stored.
        pipeline : list, optional
            A list of strings, where each string is a part of the pipeline.
            That will be used to influence or create the recommendations.
            This presence of a element will lead to the execution of this part,
            while the absence will lead force the code to skip this element.
            
            The default is ["coauthors", "coauthor_filter", "titles", "abstracts", "images"].

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """
        try:
            self.X_raw.loc[paper_id]
            
            
            # use the coauthor and coauthor_filter step
            if "coauthors" in pipeline or "coauthor_filter" in pipeline:
                # extract all co-author of the author of the paper 
                ids = self.X_raw.loc[paper_id]["author_id"]
                all_co_authors = set(np.array(list(self.author_graph.edges(ids)))[:,1])
                filters = self.X_raw_filtered["author_id"].apply(lambda x: bool(set(all_co_authors) & set(x)))
                
                # Add coauthor recommendations to the result.
                if "coauthors" in pipeline:
                    if sum(filters) > 11:
                        results["coauthors"] = self.X_raw_filtered[filters].sample(11).to_dict()
                    else:
                        results["coauthors"] = self.X_raw_filtered[filters].to_dict()
                        
                # if the coauthorfilter step is in the pipeline than limit the data to only the co-authors
                if "coauthor_filter" in pipeline:
                    # Make sure enough co-author-matches will be found to have a sufficient
                    # amount of papers for qualitative recommendations
                    # uses 11 matches because 10 are co-author-matches and 1 can be the paper itself.
                    if sum(filters) < 11:
                        results["comments"].append("Less than 10 Co-author-matches found in our dataset. "+
                                                   "To ensure our quality standards, the co-author-filter will be disabled.")
                    else:    
                        self.fit(coauthor_paper_ids=list(self.X_raw_filtered[filters].index))
            
            # if the titles-step is part of the pipeline create recommendations use the title-KNN
            if "titles" in pipeline:
                vector_title = self.vectorizer_pipeline_words.vectorize(title_text)
                results["title"] = self.get_neighbors_df(vector_title, self.neigh_titles).to_dict()
            
            # if the abstratcs-step is part of the pipeline create recommendations use the abstratcs-KNN
            if "abstracts" in pipeline:
                vector_abstracts = self.vectorizer_pipeline_words.vectorize(abstract_text)
                results["abstracts"] = self.get_neighbors_df(vector_abstracts, self.neigh_abstracts).to_dict()
            
            # if the images-step is part of the pipeline create recommendations use the images-KNN
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
        
        except:
            results["comments"].append("Paper not found in metadata, please try another paper.")
            self.fit()
            
        return results
