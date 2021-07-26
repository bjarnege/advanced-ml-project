import fitz
import io
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#ImageFile.LOAD_TRUNCATED_IMAGES = True

# customized dataset class
class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        This class creates an custom dataset based on the input data.

        Parameters
        ----------
        data : list
            The list contains all PIL images for either training or testing.
        transform : torchvision.transforms.transforms.Compose, optional
            A transforms object containing the transformation pipeline.

        Returns
        -------
        None.

        """
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        """
        Return one element of the dataset.

        Parameters
        ----------
        index : int
            index of the element to return.

        Returns
        -------
        x : array (e.g. torch.Tensor) ...
            output type depends the transforms pipeline.

        """
        x = self.data[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x
    
    def __len__(self):
        """
        Returns the dataset length.

        Returns
        -------
        int
            number of images in teh dataset.

        """
        return len(self.data)

#encoder class
class Encoder(nn.Module):
    
    def __init__(self, hidden_dim, latent_dim):
        """
        Definition of the encoder network.

        Parameters
        ----------
        hidden_dim : int
            Number of hidden dimensions in linear layer.
        latent_dim : int
            Size of the resulting latent vector.

        Returns
        -------
        None.

        """
        super(Encoder, self).__init__()

        self.Conv1 = nn.Conv2d(4, 16, 5, stride = 1)
        self.Conv2 = nn.Conv2d(16, 32, 5, stride = 1)
        self.FC_input = nn.Linear(192*192*32, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        """
        Connection of the Encoder-Layers in a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to encode.

        Returns
        -------
        mean : torch.Tensor
            The mean for the variational autoencoder.
        log_var : torch.Tensor
            The log variance for the variantional autoencoder.

        """
        h_        = F.relu(self.Conv1(x))
        h_        = F.relu(self.Conv2(h_))
        h_        = h_.view(-1, 32*192*192)
        h_       = self.LeakyReLU(self.FC_input(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        return mean, log_var

# decoder class
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        """
        Definition of the decoder network.

        Parameters
        ----------
        latent_dim : int
            Size of the used latent vector.
        hidden_dim : int
            Number of hidden dimensions in linear layer.

        Returns
        -------
        None.

        """
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, 192*192*32)
        self.decConv1 = nn.ConvTranspose2d(32, 16, 5, stride = 1)
        self.decConv2 = nn.ConvTranspose2d(16, 4, 5, stride = 1)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        """
        Connection of the Decoder-Layers in a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to decode.

        Returns
        -------
        x_hat : torch.Tensor
            The reconstructed picture.

        """
        h_     = self.LeakyReLU(self.FC_hidden(x))
        h_     = self.LeakyReLU(self.FC_output(h_))
        h_     = h_.view(-1, 32, 192, 192)
        h_     = F.relu(self.decConv1(h_))

        x_hat = torch.tanh(self.decConv2(h_))
        
        return x_hat

# auto encoder class conencting encoder and decoder with the reparametrization class in between
class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        """
        Connecting the Encoder and Decoder to create the Variational Autoencoder.

        Parameters
        ----------
        Encoder : __main__.Encoder
            A created Decoder-Object.
        Decoder : __main__.Decoder
            A created Encoder-Object.

        Returns
        -------
        None.

        """
        super(VAE, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        """
        Creating the latent Vector by Reparametrization.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the reconstrucion.
        var : torch.Tensor
            Variance of the reconstruction.

        Returns
        -------
        z : torch.Tensor
            The created latent vactor based on variance and mean.

        """
        epsilon = torch.randn_like(var).to(device)       # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        """
        Connection of the Encoder, Decoder and Reparametrization for a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input image.

        Returns
        -------
        x_hat : torch.Tensor
            The reconstructed image.
        mean : torch.Tensor
            The mean for reparamtrization for the latent vector.
        log_var : torch.Tensor
            The log-variance for reparametrization for the latent vector.
        z : torch.Tensor
            The latent vector.

        """
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)
        
        return x_hat, mean, log_var, z

# encoding function to open a pdf, read it's images, preprocess and calculate theier latent represenations and return them including its paper id
def encode(pdf_path, pdf_id, transform, model):
    """
    

    Parameters
    ----------
    pdf_path : str
        Path to the pdf file to encode.
    pdf_id : str 
        Unique arxive paper id.
    transform : torchvision.transforms.transforms.Compose
        A transforms object containing the transformation pipeline.
    model : __main__.VAE
        An created and trained variational autoencoder.

    Returns
    -------
    images_encoded : list
        List of latent vectors as numpy arrays of all pictures in the pdf.
    paper_ids : list
        List of the paper ID for all pictures in the pdf. It contains only one ID which occoures as often as the number of pictures in the file.

    """
    
    images_encoded = []
    paper_ids = []
    
    pdf_file = fitz.open(pdf_path)

    # iterate over PDF pages
    for page_index in range(len(pdf_file)):
        
        # get the page itself and search for images on it
        page = pdf_file[page_index]
        
        # iterate over found images
        for image_index, img in enumerate(page.getImageList(), start=1):
            
            try:

                # get the XREF of the image
                xref = img[0]
                # extract the image bytes
                base_image = pdf_file.extractImage(xref)
                image_bytes = base_image["image"]


                # load it to PIL, convert to RGBA and resize
                image = Image.open(io.BytesIO(image_bytes)).convert(mode="RGBA").resize((200,200))

                # proof if there is content in the picture based on the average standard deviation of the channels, ignore them otherwise
                red, green, blue, alpha = image.split()
                channel_average_standard_deviation = np.mean([np.std(np.asarray(red).flatten()), np.std(np.asarray(green).flatten()), np.std(np.asarray(blue).flatten()), np.std(np.asarray(alpha).flatten())])
                
                if channel_average_standard_deviation > 0:

                    # load image to pytorch pipeline
                    image_dataset = MyDataset([image], transform=transform)
                    image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=1)

                    # feed image to the VAE Model
                    batch_idx, data = next(enumerate(image_loader))
                    with torch.no_grad():
                        x_hat, mean, log_var, z = model(data.to(device))

                    images_encoded.append(z.cpu().numpy()[0])
                    paper_ids.append(pdf_id)
                
            except (OSError, TypeError):
                pass
            
    return images_encoded, paper_ids