import torch
from pathlib import Path
import os
from typing import Tuple

# Method for saving trainign state to a given path
# Path should be a folder
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int, path: str):
    state_dict ={
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'epoch': epoch
    }
    if not Path(path).exists():
        os.mkdir(path)
    
    # Save a state with current epoch number
    torch.save(state_dict, os.path.join(path, f"{epoch}"))
    
    # Save a copy as "latest" for easy reloading
    torch.save(state_dict, os.path.join(path, "latest"))

# Method for loading the latest checkpoint from a folder
# Path should be a folder containing state dicts
def load_latest_checkpoint(path: str) -> Tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int]:
    assert Path(path).exists()
    state_path = os.path.join(path, 'latest')
    state_dict = torch.load(state_path, weights_only=False)
    
    model = state_dict['model']
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