# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 11:40:46 2021

@author: Bjarne Gerdes
"""
import os
import torch
import requests
import numpy as np
from KNNRecommends import VAEModel as vae
import torchvision.transforms as transforms

class VectorizeData:
    
    def __init__(self, model_path, device="cpu"):
        """
        

        Parameters
        ----------
        model_path : str.
            Path to the VAE-model that will be used for image to vector transformation
        device : str, optional
            Device that will be used for image to vector transformations. The default is "cpu".

        Returns
        -------
        None.

        """
        self.device = device
        # Image transformation pipeline
        # transforms each color space of RGBA to a μ=.5 and σ=.5
        # this transformation is also used during model training and therefor necessary.
        self.transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])
                               ])
        
        # initialize instance of the VAE-encoder
        self.encoder = vae.Encoder(hidden_dim=300, latent_dim=50)
        # initialize instance of the VAE-decoder
        self.decoder = vae.Decoder(latent_dim=50, hidden_dim = 300)
        # initiliaze instance of the whole VAE
        self.VAEInstance= vae.VAE(Encoder=self.encoder, Decoder=self.decoder, device=self.device)
        # load pretrained model from model_path
        self.VAEInstance.load_state_dict(torch.load(model_path, map_location=self.device))
        self.VAEInstance.eval()
    

    def vectorize(self, url):
        """
        Is used to process a given str, containing an URL, by downloading the 
        pdf-file which can be found at that URL and extracting all images in this file.
        After that, every image will used as the anput for an varitional auto-encoder 
        and represenated as a z-vector. 
        In the final step all z-vectors will be averaged to create a vector representation
        of every image in the file.
        
        Parameters
        ----------
        url : str
            Url to the arXiv-pdf.

        Returns
        -------
        np.array
            average of the z-vector.

        """
        r = requests.get(url, allow_redirects=True)
        open('..tmp_content.pdf', 'wb').write(r.content)
        images = vae.encode('./..tmp_content.pdf', self.transform, self.VAEInstance, self.device)
        os.remove('..tmp_content.pdf')
        return np.mean(images, axis=0)
    