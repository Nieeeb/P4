import yaml
import argparse
import os
import warnings
import torch
from utils import util
import wandb
from utils.modeltools import save_checkpoint, load_or_create_state, difference
import torch.multiprocessing as mp
from datetime import timedelta
from utils.dataloader import prepare_loader
from train import setup, cleanup
import numpy as np
import cv2
from display_images import display_targets

def main():
    #Loading args from CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_file', default='utils/args.yaml', type=str)
    parser.add_argument('--world_size', default=1, type=int)

    args = parser.parse_args()

    # args for DDP
    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    print(f"Local rank: {args.local_rank}")
    print(f"World size: {args.world_size}")

    # Setting random seed for reproducability
    # Seed is 0
    util.setup_seed()

    #Loading config
    with open(args.args_file) as cf_file:
        params = yaml.safe_load( cf_file.read())
        
    # Defining world size and creating/connecting to DPP instance
    #args.local_rank = rank
    setup(args.local_rank, args.world_size)
        
    # Loading model
    # Loads if a valid checkpoint is found, otherwise creates a new model
    model, optimizer, scheduler, starting_epoch = load_or_create_state(args, params)
    
    #Dataloading Validation
    validation_loader, validation_sampler = prepare_loader(args, params,
                                file_txt=params.get('val_txt'),
                                img_folder=params.get('val_imgs'),
                                starting_epoch=starting_epoch,
                                num_workers=16,
                                chrono_difference=True
                                )
    
    difference_test(args,
                    params,
                    loader=validation_loader)
    
def difference_test(args, params, loader):
    for batchidx, (samples, targets, shapes) in enumerate(loader):
        samples = samples.cuda()
        targets = targets.cuda()
        samples = samples.float() / 255
        
        images, boxes = display_targets(samples, targets, shapes)
        for index, image in enumerate(images):
            cv2.imshow(f"{index}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if batchidx > 2:
            break
    cleanup()


if __name__ == "__main__":
    main()