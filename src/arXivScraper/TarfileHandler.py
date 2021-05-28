# -*- coding: utf-8 -*-
"""
Created on Fri May 28 17:55:03 2021

@author: Bjarne Gerdes
"""
import tarfile
import os
from pathlib import Path
import gzip
import shutil

def clean(path):
  os.remove(path)
  shutil.rmtree(path.replace(".tar", ""))
    
def untar(path, storage_path="data"):
        
    extract_path = path.replace(".tar", "")
    tar = tarfile.open(path)
    tar.extractall(extract_path)
    tar.close()        
    result_gz = ["".join([p+"/" for p in path.parts])[:-1]\
                          for path in Path(extract_path).rglob('*.gz')]
    
    if not os.path.exists(storage_path):
        os.mkdir(storage_path)
            
    output_folder = extract_path.replace("src",storage_path)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        
    # extract all tars in gz
    for file in result_gz:
        try:
            output_path = extract_path+"/"+file.split("/")[-1].replace(".gz",".tar")
            f = gzip.open(file, 'rb')
            file_content = f.read()
            f.close()
                    
            with open(output_path, "wb+") as f:
                f.write(file_content)
        except:
            if f:
                f.close()
            
    # extract all folders from tars
    folders = ["".join([p+"/" for p in path.parts])[:-1]\
                          for path in Path(extract_path).rglob('*.tar')]
        
    for f in folders:
        try:
            store_folder = output_folder+"/"+f.split("/")[-1].replace(".tar","")+"/"
            tar = tarfile.open(f)

            if not os.path.exists(store_folder):
                os.mkdir(store_folder)
                
            tar.extractall(store_folder)
            tar.close()
        except:
             if tar:
                 tar.close()
             
     