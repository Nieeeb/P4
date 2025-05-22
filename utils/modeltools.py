import torch
from pathlib import Path
import os
from typing import Tuple
from nets.nn import yolo_v8_m
from nets.autoencoder import ConvAutoencoder
from collections import OrderedDict
import cv2
from MoCo.Moco import MoCo
from MoCo.Dataset import Encoder
import copy
from Contrastive_Learner.contrastivelearner import ContrastiveLearner
from tqdm import tqdm
#from contrastive_learner.contrastive_helpers.py import load

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

    if state_dict['yolo_size'] == 'ae':
        model = ConvAutoencoder().cuda()
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
            
            if params.get("model_type") == "yolo":
                model = yolo_v8_m(len(params.get('names')))
            elif params.get("model_type") == "deep_ae":
                model = ConvAutoencoder().cuda()
            
            model = model.to(args.local_rank)
            if args.world_size > 1:
                # DDP mode
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = torch.nn.parallel.DistributedDataParallel(module=model, device_ids=[args.local_rank])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, last_epoch=-1)
            
        return model, optimizer, scheduler, starting_epoch

# Method for saving trainign state to a given path
# Path should be a folder
def save_contrastive(learner: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int, path: str):
    state_dict ={
        'learner': learner.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch
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
def load_latest_contrastive(path: str): #-> Tuple[torch.nn.Module | torch.nn.parallel.DistributedDataParallel, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int]:
    assert Path(path).exists()
    state_path = os.path.join(path, 'latest')
    state_dict = torch.load(state_path, weights_only=False)

    conv = ConvAutoencoder().cuda()
    backbone = conv.encoder
    learner = ContrastiveLearner(net = backbone,
                                image_size= 128,
                                hidden_layer=-1,
                                augment_both=True,
                                use_nt_xent_loss=True,
                                project_dim=256
                                ).cuda()
    learner.load_state_dict(state_dict=state_dict['learner'])
        
    optimizer = torch.optim.Adam(learner.parameters())
    optimizer.load_state_dict(state_dict['optimizer'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    scheduler.load_state_dict(state_dict['scheduler'])
    epoch = state_dict['epoch']
    
    return learner, optimizer, scheduler, epoch

# Given a path, first checks if a checkpoint is already saved
# If no checkpoint is found, a new model, optimizer, and scheduler are created
def load_or_create_contrastive(args):
        checkpoint_path = args['checkpoint_path']
        starting_epoch = 0
        if check_checkpoint(checkpoint_path):
            learner, optimizer, scheduler, starting_epoch = load_latest_contrastive(checkpoint_path)
            learner.to(args['local_rank'])
            print(f"Checkpoint found, starting from epoch {starting_epoch + 1}")
        else:
            print("No checkpoint found, starting new training")
            starting_epoch = 0
            
            conv = ConvAutoencoder().cuda()
            backbone = conv.encoder
            learner = ContrastiveLearner(net = backbone,
                                        image_size= 128,
                                        hidden_layer=-1,
                                        augment_both=True,
                                        use_nt_xent_loss=True,
                                        project_dim=256
                                        ).cuda()

            optimizer = torch.optim.Adam(learner.parameters())
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
            
            learner = learner.to(args['local_rank'])
            
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, last_epoch=-1)
            
        return learner, optimizer, scheduler, starting_epoch

def load_latest_clusters(path: str):
    assert Path(path).exists()
    #d = torch.device(device)
    state_path = os.path.join(path, 'latest')
    states = torch.load(state_path, weights_only=False)
    return states

def save_current_cluster(states: dict, path: str, model: torch.nn.Module, optimizer: torch.optim.Adam, scheduler: torch.optim.lr_scheduler.StepLR, epoch: int):
    current_cluster = states['current_cluster']
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch
    }
    
    states['clusters'][current_cluster] = state
    
    if not Path(path).exists():
        parent = os.path.dirname(path)
        if not Path(parent).exists():
            os.mkdir(parent)
        os.mkdir(path)
    
    # Save a state with current epoch number
    torch.save(states, os.path.join(path, f"cluster_{current_cluster}_epoch_{epoch + 1}"))
    
    # Save a copy as "latest" for easy reloading
    torch.save(states, os.path.join(path, "latest"))

def load_current_cluster(states):
    current_cluster = states['current_cluster']
    model = yolo_v8_m(num_classes=4).cuda()
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    #model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model).to(local_rank)
    model.load_state_dict(state_dict=states['clusters'][current_cluster]['model'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(state_dict=states['clusters'][current_cluster]['optimizer'])
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, last_epoch=-1)
    scheduler.load_state_dict(state_dict=states['clusters'][current_cluster]['scheduler'])
    
    epoch = states['clusters'][current_cluster]['epoch']
    
    return model, optimizer, scheduler, epoch
        
        
    states = {
        'clusters': None,
        'current_cluster': None
    }
    state = {
        'model': None,
        'optimizer': None,
        'scheduler': None,
        'epoch': None
    }
    
def load_saved_cluster_models(path: str):
    assert check_checkpoint(path)
    states = load_latest_clusters(path)
    
    models = []
    
    for n, state in tqdm(enumerate(states['clusters']), desc="Loading Models", total=len(states['clusters'])):
        states['current_cluster'] = n
        model, opt, sche, epoch = load_current_cluster(states)
        del opt, sche, epoch
        model.half()
        model.eval()
        model.cpu()
        models.append(model)
    
    return models

def load_or_create_clusters(args, params):
    n_clusters = params['n_clusters']
    checkpoint_path = params.get('checkpoint_path')
    if check_checkpoint(checkpoint_path):
        states = load_latest_clusters(checkpoint_path)
        print(f"Checkpoint found, starting from cluster {states['current_cluster']}\nStarting from epoch {states['clusters'][states['current_cluster']]['epoch'] + 1}")
    else:
        print("No checkpoint found, starting new training")
        states = {
            'clusters': [],
            'current_cluster': 0
        }
        
        base_model = yolo_v8_m(num_classes=len(params['names'])).cuda()
        base_model = torch.nn.parallel.DistributedDataParallel(base_model)
        
        base_optimizer = torch.optim.Adam(base_model.parameters(), lr=0.001)
        
        base_scheduler = torch.optim.lr_scheduler.StepLR(base_optimizer, step_size=10, last_epoch=-1)
        
        for cluster in range(n_clusters):
            state = {
            'model': copy.deepcopy(base_model.state_dict()),
            'optimizer': copy.deepcopy(base_optimizer.state_dict()),
            'scheduler': copy.deepcopy(base_scheduler.state_dict()),
            'epoch': 0
            }
            states['clusters'].append(state)
    
    return states

# Method for saving trainign state to a given path
# Path should be a folder
def save_checkpoint_dasr(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int, path: str):
    state_dict ={
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
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
def load_latest_dasr(path: str, args): #-> Tuple[torch.nn.Module | torch.nn.parallel.DistributedDataParallel, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int]:
    assert Path(path).exists()
    state_path = os.path.join(path, 'latest')
    state_dict = torch.load(state_path, weights_only=False)

    model = MoCo(Encoder,
            dim=args['dim'],
            K=args['queue_size'],
            m=args['momentum'],
            T=args['temperature']
            ).cuda()
    #model = torch.nn.parallel.DistributedDataParallel(model)
    model.load_state_dict(state_dict=state_dict['model'])
    model.to(args['local_rank'])
        
    optimizer = torch.optim.Adam(
                            model.encoder_q.parameters(),
                            lr=args['lr'], weight_decay=args['weight_decay']
                            )
    optimizer.load_state_dict(state_dict['optimizer'])
    model.to(args['local_rank'])
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], last_epoch=-1)
    scheduler.load_state_dict(state_dict['scheduler'])
    model.to(args['local_rank'])
    
    epoch = state_dict['epoch']
    
    return model, optimizer, scheduler, epoch

# Given a path, first checks if a checkpoint is already saved
# If no checkpoint is found, a new model, optimizer, and scheduler are created
def load_or_create_dasr(args):
        checkpoint_path = args['checkpoint_path']
        starting_epoch = 0
        if check_checkpoint(checkpoint_path):
            model, optimizer, scheduler, starting_epoch = load_latest_dasr(checkpoint_path, args)
            print(f"Checkpoint found, starting from epoch {starting_epoch + 1}")
        else:
            print("No checkpoint found, starting new training")
            starting_epoch = 0
            
            model = MoCo(Encoder,
                        dim=args['dim'],
                        K=args['queue_size'],
                        m=args['momentum'],
                        T=args['temperature']
                        ).cuda()
            #model = torch.nn.parallel.DistributedDataParallel(model)
            model = model.to(args['local_rank'])

            optimizer = torch.optim.Adam(
                            model.encoder_q.parameters(),
                            lr=args['lr'], weight_decay=args['weight_decay']
                            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], last_epoch=-1)
            
        return model, optimizer, scheduler, starting_epoch

def load_checkpoint_for_evaluation(args, params):
    checkpoint_path = params.get('checkpoint_path')

    assert Path(checkpoint_path).exists()
    state_path = os.path.join(checkpoint_path, params.get("best_model_epoch"))
    state_dict = torch.load(state_path, map_location='cuda:0')


    new_state_dict = OrderedDict()
    for key, value in state_dict.get("model").items():
        new_key = key[7:] 
        new_state_dict[new_key] = value

    
    
    if state_dict['yolo_size'] == 'm':
        model = yolo_v8_m(num_classes=4).cuda()
        # model = torch.nn.parallel.DistributedDataParallel(model)
        model.load_state_dict(state_dict=new_state_dict)

    optimizer = state_dict['optimizer']
    scheduler = state_dict['scheduler']
    epoch = state_dict['epoch']
    
    return model, optimizer, scheduler, epoch

def difference(bagsub, input, background):
    fgmask = bagsub.apply(input)
    output = cv2.bitwise_and(input, fgmask, mask=fgmask)

    return output