from heapq import nlargest
import random
import numpy as np
from pathlib import Path
import torch
import torchvision.transforms as transforms
import vae
from tqdm import tqdm
import pickle

# "input" is the file where the uploaded/linked comparison paper is stored
input_file = "*******.pdf"

# hyperparameters
most_similar = 0.1 # share of most similar images which are countet later
recommendation_amount = 3 # amount of papers which are recommended in the end
author_graph_shrinking = True # shrinking data based on author graph

# prepare and load the VAE model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])
                               ])

encoder = vae.Encoder(hidden_dim=300, latent_dim=50)
decoder = vae.Decoder(latent_dim=50, hidden_dim = 300)

model = vae.VAE(Encoder=encoder, Decoder=decoder).to(device)

model.load_state_dict(torch.load(f"../../resource/vae/models/VAE_epoch_178.pt", map_location=device))

model.eval()

# read the preprocessed images of the whole dataset
with open('../../resource/vae/encoded_images.pkl', 'rb') as f:
    encoding_result = pickle.load(f)

# needed to count up which paper has how often one of the n most similar images
counter = {}

# shrinking the list of images if we are building up on the author graph
if author_graph_shrinking:
    
    ##########################################################################################
    # replace following line in production by loading a stored list with all paper ids which are written by near authors
    author_graph_papers = random.sample(encoding_result['paper_ids'], round(len(encoding_result['paper_ids']) * 0.2))
    ##########################################################################################
    
    # keep only the indices of images which are also part of the shrinked papers
    paper_id_selected = [i for i, item in enumerate(encoding_result['paper_ids']) if item in set(author_graph_papers)]
    
    # filter the paper and image lists based on the remaining inices
    encoding_result['encoded_pictures'] = np.array([encoding_result['encoded_pictures'][index] for index in paper_id_selected])
    encoding_result['paper_ids'] = [encoding_result['paper_ids'][index] for index in paper_id_selected]

# count which papers are most similar based on every uploaded picture

images, ids = vae.encode(input_file, "not important", transform, model)

for image in tqdm(images):
    
    # get the index of the n most similar images (based on euclidian distance)
    best_images = np.argpartition(np.sqrt(np.sum((encoding_result['encoded_pictures'] - image)**2, axis=1)), round(len(encoding_result['encoded_pictures'])*most_similar))[:round(len(encoding_result['encoded_pictures'])*most_similar)]
    
    # count which paper is containing how often one of the most similar images
    for index in best_images:
        if encoding_result['paper_ids'][index] in counter.keys():
            counter[encoding_result['paper_ids'][index]] += 1
        else:
            counter[encoding_result['paper_ids'][index]] = 1
            
# return the k papers which hosted most often the most similar images
recommendation = nlargest(recommendation_amount, counter, key = counter.get)