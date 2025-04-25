import argparse
import os
import yaml
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import ast

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
        
    write_inference(args, params)
    #add_dates()
    
def add_dates():
    #df = pd.read_csv('DAWIDD/encodings_train_local.csv', index_col=0)
    #df['output'] = df['output'].apply(ast.literal_eval).apply(np.array)
    
    df = torch.load('DAWIDD/encodings_train_local.pickle')
    
    for index, row in tqdm(df.iterrows(), desc='Rows', total=len(df)):
        print(type(row['output']))
        print(row['output'])
        break

def write_inference(args, params):
    # data loader
    loader, _ = prepare_loader(
        args, params,
        file_txt=params['train_txt'],
        img_folder=params['train_imgs'],
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
    df.to_csv('DAWIDD/encodings_train.csv')
    torch.save(df, 'DAWIDD/encodings_train.pickle')
    print("--------- Write Complete ----------")
    
    
if __name__ == '__main__':
    main()