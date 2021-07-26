# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 13:37:46 2021

@author: Bjarne Gerdes
"""
import os

# install required packages

def download_topics(topics=["cs","physics", "quant-ph", "alg-geom"], output_path="../../resource/arxiv_data"):
    """
    Downloads arXiv-papers from google cloud bucket.

    Parameters
    ----------
    topics : list, optional
        Topics that will be downloaded. The default is ["cs","physics", "quant-ph", "alg-geom"].
    output_path : str, optional
        String where the data will be stored. The default is "../../resource/arxiv_data".

    Returns
    -------
    None.

    """
    resource_path = output_path[:output_path.rfind("/")]+"/"
    basecmd =  "gsutil -m cp -r gs://arxiv-dataset/arxiv/topic/pdf/"
   
    
    # check if the folders where the data will be stored exists.
    if not os.path.exists(resource_path):
        os.mkdir(resource_path)
        
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    
   # download each topic
    for topic in topics:
        topic_path = output_path+"/"+topic+"/"
        cmd = basecmd.replace("topic",topic)+" "+topic_path
        
        print(cmd,"\n", topic_path)
        if not os.path.exists(topic_path):
            os.mkdir(topic_path)
        
        os.system(cmd)
        
        
dt = download_topics()
