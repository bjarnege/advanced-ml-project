import pickle
import numpy as np
import pandas as pd
from flask import Flask
from flask import request
from flask_classful import FlaskView, route
from KNNRecommends.recommmender import FindNeighbors
from KNNRecommends.SciBERTVectorizer import VectorizeData as textvectorizer
from KNNRecommends.VAEVectorizer import VectorizeData as imagevectorizer

# Imitiate Flask-App
app = Flask(__name__)

def load_all_files():
    """
    This functions loads all pickle-files that are required to generate recommendations

    Returns
    -------
    fn : class
        KNN-Class that calculated recommendations.
    X_raw : pd.DataFrame
        DataFrame containing the metadata of all arxiv papers.

    """
    # load the pickle files
    author_graph = pickle.load(open("../resource/metadata/graph.pkl", "rb"))
    X_raw = pd.read_pickle("../resource/metadata/metadata_df.pkl")
    X_raw_filtered = pd.read_pickle("../resource/metadata/metadata_df_filtered.pkl")
    X_title_vectors = pd.read_pickle("../resource/transformed_data/title_vectors.pkl")
    X_abstract_vectors = pd.read_pickle("../resource/transformed_data/abstract_vectors.pkl")
    data_vae = np.load("../resource/vae_data/encoded_images.pkl", allow_pickle=True)
    X_image_vectors = pd.DataFrame(data=data_vae["encoded_pictures"], index=data_vae["paper_ids"]).groupby(data_vae["paper_ids"]).mean()
    del(data_vae)    
        
    # Initilize vectorizing pipelines
    vectorizer_pipeline_words = textvectorizer()
    vectorizer_pipeline_images = imagevectorizer("../resource/vae_data/VAE_epoch_178.pt")
    
    # Initialize the knn-class
    fn = FindNeighbors(author_graph, X_raw, X_raw_filtered, vectorizer_pipeline_words, vectorizer_pipeline_images,\
                       X_title_vectors, X_abstract_vectors, X_image_vectors, 11)
    # fit the knn models
    fn.fit()
    
    return fn, X_raw 

fn, X_raw  = load_all_files()

class API(FlaskView):
    
    def __init__(self, find_neighbors = fn, X_raw = X_raw):
        """
        This class initializes an API which can be used to generate recommendations.

        Parameters
        ----------
        find_neighbors : class, optional
            Instance of the FindNeighbors class. The default is fn.
         X_raw : pd.DataFrame
                DataFrame containing the metadata of all arxiv papers. The default is X_raw.

        Returns
        -------
        None.

        """
        self.find_neighbors = find_neighbors
        self.X_raw = X_raw
        
    
    def find_metadata(self, url):
        """
        Takes the paper url to extract the title the abstract and other metadata, 
        which will be needed to generate recommendations

        Parameters
        ----------
        url : str
            The link to the pdf-File of the paper.

        Returns
        -------
        dict
            metadata in a dict.

        """
        paper_id = url[22:].replace(".pdf","") 
        data = {"url": url,
                "paper_id": paper_id}
        
        data_metadata = self.X_raw.loc["quant-ph/0110064"][["title", "authors", "categories", "abstract"]].to_dict()
        data_metadata["abstract"] = data_metadata["abstract"].replace("\n", " ") 
        
        # merge both dicts and return them
        return {**data, **data_metadata}
    

        
    def recommend(self, url, pipeline):
        """
        

        Parameters
        ----------
        url : str
            The link to the pdf-File of the paper.
            The format of the link needs to look like the following examples:
                
                https://arxiv.org/pdf/1306.0269.pdf
                https://arxiv.org/pdf/1101.0029
                https://arxiv.org/pdf/gr-qc/9411004.pdf
                https://arxiv.org/pdf/gr-qc/9411004
                
        pipeline : str
            A comma-seperated string containing the elements of the pipeline which will be used for the recommendation.
            One can pass 5 different pipeline elements with this string:
            The possible options are:
                coauthors           ->  Will lead to recommendations based on the coauthor-graph.
                coauthor_filter     ->  Will restrict recommendations of the KNN-models (titles, abstracts, images)
                                        only to co-author papers.
                titles              ->  Will calculate recommendations based on sciBERT-Vectors of the papers title.
                abstracts           ->  Will calculate recommendations based on sciBERT-Vectors of the papers abstract.
                images              ->  Will calculate recommendations based on VAE-Vectors of the papers images.
                
            To use multiple elements the string need to follow the following format:
                coauthors,coauthor_filter,titles,abstracts,images   -> To use all 5 pipeline-elements
                titles,abstracts,images                             -> To use the KNN based models w/o the co-author-filter
                coauthor_filter, titles                             -> To created KNN based title recommendations with co-author filter
                ...
                

        Returns
        -------
        results : dict
            The recommendations of each model stored in a dict of dicts.

        """
        
        
        results = dict()
        results["comments"] = []
        
        try:
            data = self.find_metadata(url)
        except:
            results["comments"].append("Unable to find metadata. This happens most likely because "+\
                                       "the paper you've submitted is not part of our metadata, because it's too new "+\
                                       "or the submitted url is invalid. "+\
                                       "Please check the url and try an older paper.")
        
        pipeline = pipeline.replace(" ","").split(",")    
        results = self.find_neighbors.kneighbors(data["title"], data["abstract"], data["url"], data["paper_id"], 
                                            results, pipeline)
        return results

    @route('/api')
    def request_mapper(self):
        """
        Maps the requests to the right url and pipeline.
        The url parameters need to fulfill the required format defined in
        the recommend function.

        Returns
        -------
        results : dict
            The recommendations of each model stored in a dict of dicts

        """
        url = request.args.get('url')
        pipeline = request.args.get('pipeline')
        return self.recommend(url, pipeline)

if __name__ == '__main__':
    API.register(app, route_base = '/')
    app.run(host='0.0.0.0', port=12345, debug=False)


