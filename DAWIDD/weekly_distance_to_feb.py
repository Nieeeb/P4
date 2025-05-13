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
from sklearn.cluster import DBSCAN
import copy

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
    data = add_dates(args, params)
    print(data.head())
    weekly_data, february_data = group_by_week(data)
    distances_from_february = calculate_distance_from_february(weekly_data, february_data)
    list_form = [d for d in distances_from_february.values()]
    avg = sum(list_form) / len(list_form)
    
    # Print and save all distance statements to the text file
    with open('february_to_weekly_distance_1.txt', 'w') as f:
        for week, distance in distances_from_february.items():
            statement = f"Distance from February to week {week}: {distance}"
            print(statement)
            f.write(statement + "\n")
        
        avg_statement = f"Average distance from February to weeks: {avg}"
        print(avg_statement)
        f.write(avg_statement + "\n")
    
    print("Saved all distances to february_to_weekly_distance_1.txt")

def add_dates(args, params):
    df = pd.read_csv('DAWIDD/flatten_ae.csv', index_col=0)
    
    # Reconstruct 'output' column from columns 1 to 255
    output_columns = [str(i) for i in range(1, 256)]  # Columns '1' to '255'
    df['output'] = df[output_columns].apply(
        lambda row: np.array(row.values, dtype=np.float32).reshape(256, 6, 9), axis=1
    )
    
    # Load filenames from the text file
    filenames = pd.Series(get_txt(file_txt=params['train_txt'],
                                 img_folder=params['train_imgs']))
    
    datetimes = pd.Series(extract_datetimes(filenames))
    
    df['filename'] = filenames.values
    df['datetime'] = datetimes.values
    
    # Drop the individual output columns (1 to 255) to keep only 'output', 'filename', 'datetime'
    df = df.drop(columns=output_columns)
        
    return df

def flatten_output(data):
    working_data = copy.deepcopy(data)
    working_data['flat_output'] = data['output'].apply(lambda x: x.flatten())
    df = pd.DataFrame(working_data['flat_output'].tolist(), columns=[x for x in range(len(working_data.iloc[0]['flat_output']))])
    return df

def group_by_week(data):
    weekly_data = defaultdict(list)
    february_data = []
    
    for index, row in tqdm(data.iterrows(), desc="Grouping Data by Week and February", total=len(data)):
        month = row['datetime'].month
        if month == 2:  # February
            february_data.append(row['output'])
        else:
            week = row['datetime'].isocalendar().week
            weekly_data[week].append(row['output'])
        
    return weekly_data, february_data
    
def calculate_distance_from_february(weekly_data, february_data):
    # Calculate centroids for weekly data (excluding February)
    weekly_centroids = {}
    for key, data_group in tqdm(weekly_data.items(), desc='Calculating Weekly Centroids', total=len(weekly_data)):
        stacked = np.stack(data_group)  # shape: (N, 256, 6, 9)
        centroid = np.mean(stacked, axis=0)  # shape: (256, 6, 9)
        weekly_centroids[key] = centroid
    
    # Calculate centroid for all February data
    if february_data:
        february_stacked = np.stack(february_data)
        february_centroid = np.mean(february_stacked, axis=0)  # shape: (256, 6, 9)
        february_centroid_flat = february_centroid.flatten()
    else:
        raise ValueError("No data found for February to compute centroid.")
    
    # Calculate distances from February centroid to each week's centroid
    distances = {}
    for key, centroid in tqdm(weekly_centroids.items(), desc="Calculating distances from February", total=len(weekly_centroids)):
        flat_centroid = centroid.flatten()
        distance = euclidean(february_centroid_flat, flat_centroid)
        distances[key] = distance
        
    return distances

def calculate_distances(weekly_data):
    weekly_centroids = {}
    for key, data_group in tqdm(weekly_data.items(), desc='Calculating Centroids', total=len(weekly_data)):
        stacked = np.stack(data_group)  # shape: (N, 256, 6, 9)
        centroid = np.mean(stacked, axis=0)  # shape: (256, 6, 9)
        weekly_centroids[key] = centroid
    
    distances = {}
    for key, centroid in tqdm(weekly_centroids.items(), desc="Calculating distances between weeks", total=len(weekly_centroids)):
        for key1, centroid1 in weekly_centroids.items():
            if key == key1:
                continue
            if (int(key), int(key1)) in distances.keys() or (int(key1), int(key)) in distances.keys():
                pass
            flat_centroid = centroid.flatten()
            flat_centroid_1 = centroid1.flatten()
            distance = euclidean(flat_centroid, flat_centroid_1)
            distances[(int(key), int(key1))] = distance
    
    for key, dist in distances.items():
        print(f"Distance from week {key[0]} to week {key[1]}: {dist:.4f}")
    return distances

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