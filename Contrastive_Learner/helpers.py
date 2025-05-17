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
from utils.modeltools import save_checkpoint, load_or_create_state
import torch.multiprocessing as mp
from datetime import timedelta


def load_encoder(checkpoint_path: str = r'runs/feb_ae_local_2/100', 
                    nc: int = 1, 
                    nfe: int = 64, 
                    nfd: int = 64,
                    nz: int = 256,
                    device: str = 'cuda'):
    """
    Function loads encoder object from our autoencoder
    """
    model = ConvAutoencoder(nc=nc, nfe=nfe, nfd=nfd, nz=nz).to(device)
    weights = torch.load(checkpoint_path, map_location=device)
    raw = weights.get('model', weights)
    stripped = {k.replace('module.', ''): v for k, v in raw.items()}
    model.load_state_dict(stripped)
    model.eval()
    return model.encoder


def load_contrastive_learner(device, args):
    backbone = load_encoder(checkpoint_path=args['checkpoint'],
                            nc = 1,
                            nfe = 64,
                            nfd = 64,
                            nz = 256,
                            device = device
                            ) # Modify checkpoint path
    learner = ContrastiveLearner(net=backbone,
                                image_size=128,
                                hidden_layer=-1, # Get outputs of the final layer
                                project_dim=128, # producerer x-længde vector
                                use_nt_xent_loss=True,
                                temperature = 0.1,     
                                augment_both = True        
                                ).to(device)
    return learner



class ContrastiveDataset(Dataset):
    def __init__(self, files_txt: str="Data/FEBtrain.txt", img_folder: str="Data/images/train/"):
        self.paths = []
        
        with open(files_txt) as reader:
            for filename in reader.readlines():
                filename = filename.rstrip().split('/')[-1]
                self.paths.append(img_folder + filename) 
                
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((144,192))
                                            ]) #gør billeder til tensor

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.paths[idx]).convert('L'))
        return img.float() / 255