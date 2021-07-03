# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 13:37:46 2021

@author: Bjarne Gerdes
"""
import os

# install required packages

def download_topics(topics=["cs","physics", "quant-ph", "alg-geom"], output_path="../../resource/arxiv_data"):
    resource_path = output_path[:output_path.rfind("/")]+"/"
    basecmd =  "gsutil -m cp -r gs://arxiv-dataset/arxiv/topic/pdf/"
    metadatacmd = f"gsutil cp gs://arxiv-dataset/metadata-v5/arxiv-metadata-oai.json {resource_path}"

    
    if not os.path.exists(resource_path):
        os.mkdir(resource_path)
        
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    
    print("Start by downloading metadata: \n")
    os.system(metadatacmd)

    for topic in topics:
        topic_path = output_path+"/"+topic+"/"
        cmd = basecmd.replace("topic",topic)+" "+topic_path
        
        print(cmd,"\n", topic_path)
        if not os.path.exists(topic_path):
            os.mkdir(topic_path)
        
        os.system(cmd)
        
        
dt = download_topics()
