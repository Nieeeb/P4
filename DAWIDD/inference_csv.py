import argparse
import os
import yaml
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import ast
from datetime import datetime
from collections import defaultdict
from scipy.spatial.distance import euclidean

from DAWIDD_HSIC_PARRALEL_TEST import DAWIDD_HSIC
from utils.dataloader import prepare_loader
from utils import util
from nets.autoencoder import ConvAutoencoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_file', type=str, default='utils/args.yaml',
                        help="YAML file with data paths & loader params")
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--stride', type=int, default=3,
                        help="Compute HSIC every `stride` samples")
    parser.add_argument('--device', type=str, default='cuda',
                        help="torch device for encoder & GPU HSIC")
    args = parser.parse_args()

    # DDP boilerplate (if used)
    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    print(f"Local rank: {args.local_rank}, World size: {args.world_size}")

    # reproducibility
    util.setup_seed()

    # load config
    with open(args.args_file) as f:
        params = yaml.safe_load(f)
        
    #write_inference(args, params)
    add_dates(args, params)
    
def add_dates(args, params):
    #df = pd.read_csv('DAWIDD/encodings_train_local.csv', index_col=0)
    #df['output'] = df['output'].apply(ast.literal_eval).apply(np.array)
    
    df = torch.load('DAWIDD/encodings_train.pickle')
    
    filenames = pd.Series(get_txt(file_txt=params['train_txt'],
                        img_folder=params['train_imgs']))
    
    datetimes = pd.Series(extract_datetimes(filenames))
    
    df['filename'] = filenames.values
    
    df['datetime'] = datetimes.values
    
    monthly_data = defaultdict(list)
    for index, row in tqdm(df.iterrows(), desc="Grouping Data by Month", total=(len(df))):
        month = row['datetime'].month
        monthly_data[month].append(row['output'])
    
    monthly_centroids = {}
    for key, data_group in tqdm(monthly_data.items(), desc='Calculating Centroids', total=len(monthly_data)):
        stacked = np.stack(data_group)  # shape: (N, 256, 6, 9)
        centroid = np.mean(stacked, axis=0)  # shape: (256, 6, 9)
        monthly_centroids[key] = centroid
    
    baseline_month = 2
    baseline_centroid = monthly_centroids[baseline_month].flatten()
    
    distances = {}
    for key, centroid in tqdm(monthly_centroids.items(), desc="Calculating distances from baseline month", total=len(monthly_centroids)):
        if key == baseline_month:
            continue
        flat_centroid = centroid.flatten()
        distance = euclidean(baseline_centroid, flat_centroid)
        
        distances[key] = distance
        
    for key, dist in distances.items():
        print(f"Distance from baseline month {baseline_month} to month {key}: {dist:.4f}")

def write_inference(args, params):
    # data loader
    loader, _ = prepare_loader(
        args, params,
        file_txt=params['val_txt'],
        img_folder=params['val_imgs'],
        starting_epoch=-1,
        num_workers=16,
        shuffle = False
    )

    # checkpoint path
    ckpt = '/ceph/project/DAKI4-thermal-2025/P4/runs/ae_complex_full_1/100'
    #ckpt = '/home/nieb/Projects/DAKI Projects/P4/DAWIDD/ae_complex'
    
    device = torch.device(args.local_rank)
    model = ConvAutoencoder(nc=1, nfe=64, nfd=64, nz=256).to(device)
    ckpt = torch.load(ckpt, map_location=device)
    raw = ckpt.get('model', ckpt)
    stripped = {k.replace('module.', ''): v for k, v in raw.items()}
    model.load_state_dict(stripped)
    model.eval()
    
    encodings = []
    with torch.no_grad():
        for images, *_ in tqdm(loader, total=len(loader), desc="Batches"):
            images = images.to(args.device).float() / 255
            outputs = model.encode(images)
            
            for output in outputs:
                encoding = {
                    'output': output.cpu().numpy()
                }
                encodings.append(encoding)
    
    df = pd.DataFrame(encodings)
    #df.to_csv('DAWIDD/encodings_valid_local.csv')
    torch.save(df, 'DAWIDD/encodings_valid_local.pickle')
    print("--------- Write Complete ----------")
    
def get_txt(file_txt, img_folder):
    filenames = []
        
    with open(file_txt) as reader:
        lines = reader.readlines()
        for filename in tqdm(lines, desc='Locating File Names', total=len(lines)):
            filename = filename.rstrip().split('/')[-1]
            filenames.append(img_folder + filename)
    
    return filenames

def extract_datetimes(filenames):
    datetimes = []
    for filename in tqdm(filenames, desc='Extracting Datetimes', total=len(filenames)):
        date_str = filename.split('_')[0].split('/')[-1]
        time_str = filename.split('_')[3]
        datetime_str = date_str + time_str
        
        datetime_obj = datetime.strptime(datetime_str, r'%Y%m%d%H%M')
        datetimes.append(datetime_obj)
        
    return datetimes

if __name__ == '__main__':
    main()