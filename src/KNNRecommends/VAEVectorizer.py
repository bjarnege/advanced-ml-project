# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 11:40:46 2021

@author: Bjarne Gerdes
"""
import torch
import torchvision.transforms as transforms
import VAEModel as vae
import numpy as np
import requests
import os

class VectorizeData:
    
    def __init__(self, model_path="../../resource/vae_data/VAE_epoch_178.pt", device="cpu"):
        
        self.device = device
        self.transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])
                               ])
        
        self.encoder = vae.Encoder(hidden_dim=300, latent_dim=50)
        self.decoder = vae.Decoder(latent_dim=50, hidden_dim = 300)
        self.VAEInstance= vae.VAE(Encoder=self.encoder, Decoder=self.decoder, device=self.device)
        self.VAEInstance.load_state_dict(torch.load(model_path, map_location=self.device))
        self.VAEInstance.eval()
    

    def vectorize(self, url):
        r = requests.get(url, allow_redirects=True)
        open('..tmp_content.pdf', 'wb').write(r.content)
        images = vae.encode('./..tmp_content.pdf', self.transform, self.VAEInstance, self.device)
        os.remove('..tmp_content.pdf')
        return np.mean(images, axis=0)
    