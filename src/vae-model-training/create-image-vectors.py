import os
from google_drive_downloader import GoogleDriveDownloader as gdd
import torch
import torchvision.transforms as transforms
import vae
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import pickle



# download the model and encodings if not already excisting
if list(Path("../../resource/vae_data/").rglob("*.*")) == []:
    os.mkdir("../../resource/vae_data/")

    gdd.download_file_from_google_drive(file_id='1BEIoJD61XybT9Vhq3pW6OEWYTWE48PPZ', dest_path='../../resource/vae_data/epoch147.zip', unzip=True)
    os.remove('../../resource/vae_data/epoch147.zip')

    gdd.download_file_from_google_drive(file_id='1VMzrhLfM_ltsKQs1btaZijfBHoXJcOCJ', dest_path='../../resource/vae_data/encoded_images.zip', unzip=True)
    os.remove('../../resource/vae_data/encoded_images.zip')

# function to read all paper ids and filepaths to a dataframe
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

# prepare and load the VAE model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])
                               ])

encoder = vae.Encoder(hidden_dim=300, latent_dim=50)
decoder = vae.Decoder(latent_dim=50, hidden_dim = 300)

model = vae.VAE(Encoder=encoder, Decoder=decoder).to("cpu")

model.load_state_dict(torch.load("../../resource/vae_data/VAE_epoch_147.pt", map_location="cpu"))

model.eval()

model = model.to(device)
# initialize dict to save the encoded pictures and its paper IDs
encoding_result = {'encoded_pictures': [], 'paper_ids': []}

# read all filenames and paper ids
files = read_all_filename()

# iterate over papers
for idx in tqdm(list(files.index)):
    
    # load paper, extract images, resize them and calculate latent space
    images, ids = vae.encode(files.loc[idx][0], idx, transform, model)
    
    # save latent space and paper id of each image
    encoding_result['encoded_pictures'] += images
    encoding_result['paper_ids'] += ids


with open("../../resource/vae_data/encoded_images.pkl", 'wb') as f:
    pickle.dump(encoding_result, f)



