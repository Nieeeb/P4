import sys
import os

root_dir = os.getcwd()
sys.path.append(root_dir)

import json
from minio import Minio
import urllib3
from typing import Tuple, Dict
#from Dataloading.coco import COCODataset
from torch.utils.data import DataLoader
from PIL import Image
import io

# Default method for loading credentials.
# Expects a file secrets/credentials.json in the root directory
# File must contain: {"endpoint": "xxx", "bucket": "xxx", "username": "xxx", "password": "xxx"}
def load_credentials() -> Dict[str, str]:
    path = r'secrets/credentials.json'
    stupid_path = r'/home/nieb/Projects/DAKI Mini Projects/MLOps/secrets/credentials.json'
    assert os.path.exists(stupid_path)
    
    credentials = None
    with open(stupid_path, 'r', encoding='utf-8') as file:
        credentials = json.load(file)
        file.close()
    return credentials

# Method that opens the session to them Minio storage bucket
def create_session() -> Tuple[urllib3.response.HTTPResponse, Dict[str, str]]:
    credentials = load_credentials()
    
    client = Minio(
        endpoint=credentials['endpoint'],
        access_key=credentials['username'],
        secret_key=credentials['password'],
    )
    #Stripping secret information
    credentials['username'] = None
    credentials['password'] = None
    
    return client, credentials

# Downloads a specified boject into a folder as a file
# Skips downloading if a file of the same name already exists
def download_file(client, bucket, object_path, output_path):
    pass
    # Get data of an object.
    response = None
    try:
        # Checking if file already exists
        if not os.path.exists(output_path):
            print("Creating empty temporary file...")
            open(output_path, 'w').close()
        else:
            print(f"File already exists in {output_path}, skipping download...")
            return
        # Requesting object from storage bucket
        print("Getting response...")
        response = client.get_object(bucket, object_path)
        # Read data from response.
        with open(output_path, 'wb') as file:
            print("Writing file...")
            file.write(response.data)
            file.close()
            print(f"File written to {output_path}")
    finally:
        if response:
            response.close()
            response.release_conn()

# Streams an object from the bucket
# The object is returned as a HTTPresponse, and must be decoded before use
def stream_file(client, bucket, object_path) -> urllib3.response.HTTPResponse:
    response = None
    try:
        response = client.get_object(bucket, object_path)
        # print(type(response))
        data = response.data
        # print(type(data))
        return data
    finally:
        if response:
            response.close()
            response.release_conn()

# Streams an image to be used
# Does not save image as a file
# Decodes the bytestream to a PIL image
def request_image(client, bucket, image_path) -> Image:
    data = stream_file(client, bucket, image_path)
    image = Image.open(io.BytesIO(data))
    return image

def main():
    test_img = "harborfrontv2/frames/20200514/clip_21_2239/image_0106.jpg"
    client, credentials = create_session()
    object = 'Valid.json'
    object_path = 'harborfrontv2/' + object
    output_path = 'outputs/' + object
    download_file(client, credentials['bucket'], object_path, output_path)
    
    # Under construction
    #train_dataset, valid_dataset, test_dataset = load_online_datasets(None)
    
    image = request_image(client, credentials['bucket'], test_img)
    image.show()
    
    #test_dataset = COCODataset(root='', annotation=output_path, numClass=4, online=True)
    #print(len(test_dataset))
    #dataloader = DataLoader(test_dataset, batch_size=1)
    
    #for iteration, (x, y) in enumerate(dataloader):
    #    print(x)
    #    print(y)

if __name__ == '__main__':
    main()