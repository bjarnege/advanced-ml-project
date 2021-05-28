from tqdm import tqdm
from bs4 import BeautifulSoup
from XMLReader import XMLParser
import boto3, configparser, os, botocore
from TarfileHandler import untar, clean

s3resource = None

class ArXivS3Scraper:
    
    def setup(self):
        """Creates S3 resource & sets configs to enable download."""
    
        print('Connecting to Amazon S3...')
    
        # Securely import configs from private config file
        configs = configparser.SafeConfigParser()
        configs.read('config.ini')
    
        # Create S3 resource & set configs
        global s3resource
        s3resource = boto3.resource(
            's3',  # the AWS resource we want to use
            aws_access_key_id=configs['DEFAULT']['ACCESS_KEY'],
            aws_secret_access_key=configs['DEFAULT']['SECRET_KEY'],
            region_name='us-east-1'  # same region arxiv bucket is in
        )
        
        if not os.path.exists("data"):
            os.mkdir("data")

    def download_file(self, key, pbar=None):
        """
        Downloads given filename from source bucket to destination directory.
        Parameters
        ----------
        key : str
            Name of file to download
        """
        filename = key.split("/")[-1]
        # Ensure src directory exists 
        if not os.path.isdir('src'):
            os.makedirs('src')
    
        # Download file
        if type(pbar) == type(None):
            print('\nDownloading s3://arxiv/{} to {}...'.format(key, key))
        else:
            pbar.set_description('\nDownloading s3://arxiv/{} to {}...'.format(key, key))
            
        try:
            s3resource.meta.client.download_file(
                Bucket='arxiv', 
                Key=key,  # name of file to download from
                Filename= ("src/"+filename),  # path to file to download to
                ExtraArgs={'RequestPayer':'requester'})
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print('ERROR: ' + key + " does not exist in arxiv bucket")
    
    def explore_metadata(self, metadata_path):
        """Explores arxiv bucket metadata."""
    
        print('\narxiv bucket metadata:')
    
        with open(metadata_name, 'r') as manifest:
            soup = BeautifulSoup(manifest, 'xml')
    
            # Print last time the manifest was edited
            timestamp = soup.arXivSRC.find('timestamp', recursive=False).string
            print('Manifest was last edited on ' + timestamp)
    
            # Print number of files in bucket
            numOfFiles = len(soup.find_all('file'))
            print('arxiv bucket contains ' + str(numOfFiles) + ' tars')
    
            # Print total size
            total_size = 0
            for size in soup.find_all('size'):
                total_size = total_size + int(size.string)
            print('Total size: ' + str(round(total_size/1000000000,3)) + ' GB')
    
        print('')

        parser = XMLParser(metadata_path)
        self.manifest_df = parser.xml_to_pandas_df()
        self.num_tars = self.manifest_df.shape[0]
    

    def process(self, every_n_th_file=8):
        """Sets up download of tars from arxiv bucket."""
    
        print('Beginning tar download & extraction...')
    
        # Create a reusable Paginator
        paginator = s3resource.meta.client.get_paginator('list_objects_v2')
    
        # Create a PageIterator from the Paginator
        page_iterator = paginator.paginate(
            Bucket='arxiv',
            RequestPayer='requester',
            Prefix='src/'
        )
    
        # Download and extract tars
        numFiles = 0
        i = 0
        pbar = tqdm(total=self.num_tars)
        
        for page in page_iterator:
                numFiles = numFiles + len(page['Contents'])
                for file in page['Contents']:
                    i += 1
                    # download every 10th zip file:
                    if i % every_n_th_file == 0:
                        key = file['Key']
                        # If current file is a tar
                        if key.endswith('.tar'):
                            self.download_file(key, pbar)
                    
                            # extract data
                            untar(key)
                            clean(key)
                            
                    pbar.update(1)
                    
        print('Processed ' + str(numFiles - 1) + ' tars')  # -1 

    
metadata_name = "src/arXiv_src_manifest.xml"

if __name__ == '__main__':
    """Runs if script is called on command line"""
    ax = ArXivS3Scraper()
    # Create S3 resource & set configs
    ax.setup()

    # Download manifest file to current directory
    ax.download_file(metadata_name)

    # Explore bucket metadata 
    ax.explore_metadata(metadata_name)

    # Begin tar download & extraction 
    ax.process()
