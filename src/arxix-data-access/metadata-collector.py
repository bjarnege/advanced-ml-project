# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 13:50:36 2021

@author: Bjarne Gerdes
"""
import os
import json 
import pickle
import pandas as pd
from tqdm import tqdm
from pathlib import Path



def read_all_metadata(metadata_path = "../../resource/arxiv-metadata-oai-snapshot.json"):
    articles = []
    with open(metadata_path, "r") as f:
        for l in tqdm(f):
            d = json.loads(l)
            data = {"id": d["id"],"title": d["title"], "authors": d['authors'],
                    "categories": d["categories"], 'abstract': d['abstract'],
                   "doi": d["doi"], "authors_parsed": d["authors_parsed"]}
            articles.append(data)    
    return pd.DataFrame(articles).set_index("id")

def read_all_filename(output_path = "../../resource/arxiv_data"):
    paths_check = []
    paths = []
    indexes = []
    for path in list(Path(output_path).rglob('*.pdf'))[::-1]:
        path = str(path).replace("\\", "/")
        path_w_o_version = "".join(path.split("v")[:-1])
        if not path_w_o_version in paths_check:
            paths_check.append(path_w_o_version)
            paths.append(path)
            
            # extract id from path
            data = path[len(output_path):].split("/")
            category = data[1]
            id_cat = data[-1].split("v")[0]
            idx = category + "/" + id_cat
            indexes.append(idx)
            
    df_files = pd.DataFrame({"id": indexes, "filepath": paths})
    return df_files.set_index("id")

def filter_by_filenames(df_filenames, df_metadata):
    df_metadata_filtered = df_metadata\
        [df_metadata.index.isin(df_filenames.index)]
        
    return df_metadata_filtered.merge(df_filenames, left_index=True, right_index=True)
        
def author_id_mapping(df_metadata):
    all_authors = list()
    df_metadata["authors_parsed"].apply(lambda x: [all_authors.append(author) for author in x])
    id_dict = dict()
    
    all_authors = set(["".join(a) for a in all_authors])
    for idx, author in enumerate(all_authors):
        id_dict[author] = idx 

    return id_dict

def author_id_lookup(author, id_dict):
    author_str = "".join(author)
    
    return id_dict[author_str]
    
#%% Read all existing files and find their metadata
# + map authors to a specific id
df_filenames = read_all_filename()
df_metadata = read_all_metadata()
df_metadata_filtered = filter_by_filenames(df_filenames, df_metadata)
id_dict = author_id_mapping(df_metadata)
df_metadata_filtered["author_id"] = df_metadata_filtered["authors_parsed"].apply(lambda x: [author_id_lookup(author, id_dict) for 
                                                   author in x])

#%% Store files as pickle-Objects
metadata_path = "../../resource/metadata/"
if not os.path.exists(metadata_path):
    os.mkdir(metadata_path)
    
df_metadata_filtered.to_pickle(metadata_path+"/metadata_df_filtered.pkl")

with open(metadata_path+"/id_mapping_dict.pkl", 'wb+') as handle:
        pickle.dump(id_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
