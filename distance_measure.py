import yaml
import argparse
import os
import warnings
import torch
from utils import util
import wandb
from utils.modeltools import save_checkpoint, load_or_create_state
import torch.multiprocessing as mp
from datetime import timedelta
from utils.dataloader import prepare_loader
import cv2
import numpy as np

warnings.filterwarnings("ignore")


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
    
    # Loading model
        # Loads if a valid checkpoint is found, otherwise creates a new model
    # model, optimizer, scheduler, starting_epoch = load_or_create_state(args, params)
    
    #Dataloading train
    #train_loader, train_sampler = prepare_loader(args, params,
    #                                file_txt=params.get('train_txt'),
    #                                img_folder=params.get('train_imgs'),
    #                                starting_epoch=starting_epoch
    #                                )
    
    #Dataloading Validation
    validation_loader, validation_sampler = prepare_loader(args, params,
                                    file_txt=params.get('val_txt'),
                                    img_folder=params.get('val_imgs'),
                                    starting_epoch=-1
                                    )
    # Iterates through validation set
    # Disables gradient calculations
    with torch.no_grad():
        for _, (samples, targets, _) in enumerate(validation_loader):
            # Sending data to appropriate GPU
            samples, targets = samples.to(args.local_rank), targets.to(args.local_rank)
            image = None
            #print(targets[24].shape)
            #print(samples[24].shape)
            for sample_index, sample in enumerate(samples):
                matches = []
                for target_index, target in enumerate(targets):
                    #print(target[0].item)
                    if target[0].item() == sample_index:
                        matches.append(target)
                
                sample = sample.float() / 255
                
                image = np.transpose(sample.cpu().numpy(), (1, 2, 0))
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                img_width = 288
                img_height = 384
                sample = sample.cpu()
                for i, match in enumerate(matches):
                    center_x = match[2].item()
                    center_y = match[3].item()
                    box_width = match[4].item()
                    box_height = match[5].item()
                    
                    top_x = int((center_x + (box_width / 2)) * img_height)
                    bottom_x = int((center_x - (box_width / 2)) * img_height)
                    top_y = int((center_y - (box_height / 2)) * img_width)
                    bottom_y = int((center_y + (box_height / 2)) * img_width)
                    print(top_x)
                    #print(int(match[2].item()*width))
                    cv2.rectangle(image, (top_x, top_y), (bottom_x, bottom_y), (255, 0, 0), 1)
                
                #cv2.imshow("image", image)
                #break
                #print(f"{matches} in {sample_index}")
                #print(sample.shape)
            #image = np.transpose(samples[17].cpu().numpy(), (1, 2, 0))
            cv2.imshow("image", image)
            break
            
            #samples = samples.float() / 255 # Input images are 8 bit single channel images. Converts to 0-1 floats
            
            #outputs = model(samples) # Forward pass
            
            #vloss = criterion(outputs, targets) # Calculating loss
            
            #torch.distributed.reduce(vloss, torch.distributed.ReduceOp.AVG) # Syncs loss and takes the average across GPUs
            #v_loss.update(vloss.item(), samples.size(0))
            
            #del outputs
            #del vloss
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()