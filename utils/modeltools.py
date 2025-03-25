import torch
from pathlib import Path
import os
from typing import Tuple
from nets.nn import yolo_v8_m
from collections import OrderedDict

# Method for saving trainign state to a given path
# Path should be a folder
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int, path: str, yolo_size: str):
    state_dict ={
        'model': model.state_dict(),
        'optimizer': optimizer,
        'scheduler': scheduler,
        'epoch': epoch,
        'yolo_size': yolo_size
    }
    if not Path(path).exists():
        parent = os.path.dirname(path)
        if not Path(parent).exists():
            os.mkdir(parent)
        os.mkdir(path)
    
    # Save a state with current epoch number
    torch.save(state_dict, os.path.join(path, f"{epoch}"))
    
    # Save a copy as "latest" for easy reloading
    torch.save(state_dict, os.path.join(path, "latest"))

# Method for loading the latest checkpoint from a folder
# Path should be a folder containing state dicts
def load_latest_checkpoint(path: str): #-> Tuple[torch.nn.Module | torch.nn.parallel.DistributedDataParallel, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int]:
    assert Path(path).exists()
    state_path = os.path.join(path, 'latest')
    state_dict = torch.load(state_path, weights_only=False)
    
    if state_dict['yolo_size'] == 'm':
        model = yolo_v8_m(num_classes=4).cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
        model.load_state_dict(state_dict=state_dict['model'])

    optimizer = state_dict['optimizer']
    scheduler = state_dict['scheduler']
    epoch = state_dict['epoch']
    
    return model, optimizer, scheduler, epoch

# Method that checks if checkpoint files exists in a folder
# Returns False if no latest checkpoint is found
def check_checkpoint(path: str) -> bool:
    latet_path = os.path.join(path, 'latest')
    
    if os.path.isfile(latet_path):
        return True
    else:
        return False

# Given a path, first checks if a checkpoint is already saved
# If no checkpoint is found, a new model, optimizer, and scheduler are created
def load_or_create_state(args, params):
        checkpoint_path = params.get('checkpoint_path')
        starting_epoch = 0
        if check_checkpoint(checkpoint_path):
            model, optimizer, scheduler, starting_epoch = load_latest_checkpoint(checkpoint_path)
            model.to(args.local_rank)
            print(f"Checkpoint found, starting from epoch {starting_epoch + 1}")
        else:
            print("No checkpoint found, starting new training")
            starting_epoch = 0
            model = yolo_v8_m(len(params.get('names')))
            model = model.to(args.local_rank)
            if args.world_size > 1:
                # DDP mode
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = torch.nn.parallel.DistributedDataParallel(module=model, device_ids=[args.local_rank])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, last_epoch=-1)
            
        return model, optimizer, scheduler, starting_epoch



def load_checkpoint_for_evaluation(args, params):
    checkpoint_path = params.get('checkpoint_path')

    assert Path(checkpoint_path).exists()
    state_path = os.path.join(checkpoint_path, params.get("best_model_epoch"))
    state_dict = torch.load(state_path, map_location='cuda:0')


    new_state_dict = OrderedDict()
    for key, item in state_dict.items():
        print(key)
    
    
    if state_dict['yolo_size'] == 'm':
        model = yolo_v8_m(num_classes=4).cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
        model.load_state_dict(state_dict=state_dict['model'])

    optimizer = state_dict['optimizer']
    scheduler = state_dict['scheduler']
    epoch = state_dict['epoch']
    
    return model, optimizer, scheduler, epoch