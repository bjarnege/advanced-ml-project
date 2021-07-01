from flask import Flask
import pandas as pd
import json
import pickle
from flask_classful import FlaskView, route
from KNNRecommends.Vectorizer import VectorizeData
from KNNRecommends.recommmender import FindNeighbors


# Imitiate Flask-App
app = Flask(__name__)
base_path = "../resource/"

class API(FlaskView):
    
    def __init__(self, metadata_filtered = base_path+"metadata/metadata_df_filtered.pkl",\
                 metadata_full = base_path+"arxiv-metadata-oai-snapshot.json",\
                 abstract_vectors = base_path+"transformed_data/abstract_vectors.pkl",\
                 title_vectors = base_path+"transformed_data/title_vectors.pkl",\
                 author_id_mapping = base_path+"metadata/id_mapping_dict.pkl"):
        
        self.metadata_filtered = pd.read_pickle(metadata_filtered)
        self.metadata_full = self.__extract_metadata_full(metadata_full)
        
        with open(author_id_mapping, "rb") as mapping: 
            self.author_id_mapping = pickle.load(mapping)
            
        # Initialize KNN BERT-Models
        self.abstract_vectors = pd.read_pickle(abstract_vectors)
        self.title_vectors = pd.read_pickle(title_vectors)
        
        self.knn_title = FindNeighbors(self.metadata_filtered, VectorizeData(), self.title_vectors)
        self.knn_abstract = FindNeighbors(self.metadata_filtered, VectorizeData(), self.abstract_vectors)    
        
        self.knn_title.fit(k=10)
        self.knn_abstract.fit(k=10)
            
        
    def __extract_metadata_full(self, metadata_full):
        articles = []
        with open(metadata_full, "r") as f:
            for l in f:
                d = json.loads(l)
                data = {"id": d["id"],"title": d["title"], "authors": d['authors'],
                    "categories": d["categories"], 'abstract': d['abstract'],
                   "doi": d["doi"], "authors_parsed": d["authors_parsed"]}
                articles.append(data)    
            
        return pd.DataFrame(articles).set_index("id")
        
    
    def __find_paper_data(self, url):
        paper_id = url.replace("https://arxiv.org/abs/","")
        data =  self.metadata_full.loc[paper_id]
        data["author_ids"] = [self.author_id_mapping["".join(author)] for author in data["authors_parsed"]]

        return data.to_dict()
    
    @route('/api/<url>')
    def recommend(self, url):
        data = self.__find_paper_data(url)
        print(data)
        data["top_n_titles"] = self.knn_title.kneighbors(data["title"])["cos distance"].to_dict()
        data["top_n_abstracts"] = self.knn_abstract.kneighbors(data["title"])["cos distance"].to_dict()
        
        return data


if __name__ == '__main__':
    API.register(app, route_base = '/')
    app.run(host='0.0.0.0', port=5050, debug=False)
