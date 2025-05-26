import torch
from Contrastive_Learner.contrastivelearner import ContrastiveLearner
from nets.autoencoder import ConvAutoencoder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from utils import util
import yaml
import os
import argparse
import warnings
import torch
import wandb
from utils.modeltools import save_contrastive, load_or_create_contrastive
import torch.multiprocessing as mp
from datetime import timedelta
from train import setup
from Contrastive_Learner.helpers import ContrastiveDataset
from tqdm import tqdm

def train_learner(args):
    device = args['local_rank']
    learner, optimizer, scheduler, starting_epoch = load_or_create_contrastive(args)
    
    learner.to(device)
    
    dataset = ContrastiveDataset(files_txt = args['train_txt'],
                                img_folder = args['train_imgs']
                                )
    
    data_loader = DataLoader(dataset,
                            batch_size=args['batch_size'],
                            shuffle=True,
                            num_workers=args['num_workers'])
    if args['local_rank'] == 0:
        print("Starting training")
    for epoch in range(starting_epoch, args['epochs']):
        if args['local_rank'] == 0:
            print(f"Training epoch {epoch + 1}")
        m_loss = util.AverageMeter()
        learner.train()
        for batchidx, images in (pbar := tqdm(enumerate(data_loader), total=len(data_loader))):
            images = images.to(device)
            optimizer.zero_grad()
            loss = learner(images)
            loss.backward()
            optimizer.step()
            
            m_loss.update(loss.item(), images.size(0))
            
            pbar.set_description(f"{epoch + 1}/{args['epochs']} Loss: {loss:.4f}")
            
            # Logging to wandb
            if args['local_rank'] == 0:
                e = epoch * len(data_loader) + 1
                s = batchidx
                step = e + s
                wandb.log({
                "Training step": step,
                "Training mloss average": m_loss.avg,
                "Raw loss": loss
                })
                
            del loss
            
            if batchidx > 10:
                break
        
        scheduler.step()
        
        if args['local_rank'] == 0:
            print(f"Training for for epoch {epoch  + 1} complete. Average loss is at {m_loss.avg}")
            wandb.log({
                'Epoch': epoch + 1,
                'Training Epoch Loss': m_loss.avg
            })
        
        del m_loss
        
        if args['local_rank'] == 0:
            print(f"Saving Checkpoint for epoch {epoch + 1}")
            save_contrastive(learner, optimizer, scheduler, epoch + 1, args['checkpoint_path'])

def train(rank, args):
        # Init Wandb
    if args['local_rank'] == 0:
        wandb.init(
            project="Thermal",
            config=args,
            resume="allow",
            group=args['run_name'],
            id=args['run_name']
        )

    train_learner(args)

def main():
    #Loading args from CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_file', default='utils/contrastive_args.yaml', type=str)
    parser.add_argument('--world_size', default=1, type=int)

    args = parser.parse_args()

    # args for DDP
    args.local_rank = int(os.getenv('LOCAL_RANK', 0))

    # Setting random seed for reproducability
    # Seed is 0
    util.setup_seed()

    #Loading config
    with open(args.args_file) as cf_file:
        params = yaml.safe_load( cf_file.read())
        
    params['local_rank'] = args.local_rank
    params['world_size'] = args.world_size
    params['args_file'] = args.args_file
    
    print(f"Local rank: {params['local_rank']}")
    print(f"World size: {params['world_size']}")
    
    # Creating training instances for each GPU
    #mp.spawn(train, args=(args, params), nprocs=args.world_size, join=True)
    
    train(args.local_rank, params)

if __name__ == "__main__":
    main()
