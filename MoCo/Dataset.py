from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import os
from random import random
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import torchvision
import torch
#from utils.util import setup_seed
from torch import nn
import torch.nn.functional as F
import numpy
import argparse
import yaml
from MoCo.Moco import MoCo
import copy
from torchvision.models import resnet18
from collections import OrderedDict

def load_args():
    #Loading args from CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_file', default='utils/dasr_args.yaml', type=str)
    parser.add_argument('--world_size', default=1, type=int)

    args = parser.parse_args()

    # args for DDP
    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    print(f"Local rank: {args.local_rank}")
    print(f"World size: {args.world_size}")

    #Loading config
    with open(args.args_file) as cf_file:
        params = yaml.safe_load( cf_file.read())
        
    return params

def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    #benchmark gør så den finder den bedste algoritme (SKAL ALTID VÆRE FALSE)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class GrayscalePatchDataset(Dataset):
    def __init__(self, files_txt: str="Data/train.txt", img_folder: str="Data/images/train/", patch_size: int=256):
        self.paths = []
        
        with open(files_txt) as reader:
            for filename in reader.readlines():
                filename = filename.rstrip().split('/')[-1]
                self.paths.append(img_folder + filename) 
                
        self.patch_size = patch_size
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.RandomRotation(45)
                                            ]) #gør billeder til tensor

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        W, H = img.size
        ps = self.patch_size
        
        x1, x2 = self.rand_crop(img, W, H, ps), self.rand_crop(img, W, H, ps)
        return x1, x2, self.paths[idx]
    
    def rand_crop(self, img, W, H, ps):
        x = random.randint(0, W-ps)
        y = random.randint(0, H-ps)
        return self.transform(img.crop((x, y, x+ps, y+ps)))
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return fea, out

def plot_patch_pairs(loader):
    # Grab a single batch
    x1_batch, x2_batch, paths = next(iter(loader))
    batch_size = x1_batch.size(0)

    # Create a figure with batch_size rows and 2 columns
    fig, axes = plt.subplots(batch_size, 2, figsize=(6, batch_size * 3))
    if batch_size == 1:
        axes = axes.reshape(1, 2)  # ensure 2D indexing for single-sample batch

    for i in range(batch_size):
        # x1
        ax1 = axes[i, 0]
        ax1.imshow(x1_batch[i].squeeze().cpu(), cmap='gray')
        ax1.set_title(f"Sample {i} – x1")
        ax1.axis('off')

        # x2
        ax2 = axes[i, 1]
        ax2.imshow(x2_batch[i].squeeze().cpu(), cmap='gray')
        ax2.set_title(f"Sample {i} – x2")
        ax2.axis('off')

    plt.tight_layout()
    plt.show()

# Function for defining machine and port to use for DDP
# Sets up the process group
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    #os.environ['CUDA_VISIBLE_DEVICES'] = os.getenv('LOCAL_RANK', 0)
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


# Method to gracefully destory the process group
# If this is not done, problems may arise on future runs
def cleanup():
    torch.distributed.destroy_process_group()

def main():
    args = load_args()
    setup_seed()
    setup(0, 1)
    dataset = GrayscalePatchDataset(patch_size=args['patch_size'])
    sampler = DistributedSampler(dataset=dataset, shuffle=True, drop_last=False)
    sampler.set_epoch(-1)
    loader = DataLoader(dataset, batch_size=args['batch_size'], sampler=sampler, drop_last=False, pin_memory=True)

    moco_model = MoCo(Encoder,
                dim=256, K=args['queue_size'],
                m=args['momentum'], T=args['temperature'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    moco_model = moco_model.to(device)

    num_epochs = 10
    loss_fn = torch.nn.CrossEntropyLoss().cuda()    
    
    optimizer = torch.optim.Adam(
        moco_model.module.encoder_q.parameters() if isinstance(moco_model, nn.DataParallel) 
        else moco_model.encoder_q.parameters(),
        lr=args['lr'], weight_decay=1e-4
    )
    #plot_patch_pairs(loader)
    for epoch in range(num_epochs):
        moco_model.train()
        sampler.set_epoch(epoch)
        for q, k, _ in loader:
            optimizer.zero_grad()
            q = q.to(device)
            k = k.to(device)

            embeddings, logits, labels = moco_model(q, k)
            
            loss = loss_fn(logits, labels)
            
            loss.backward()
            
            optimizer.step()

            print(f"Epoch: {epoch +1}, loss: {loss.item()}")

    cleanup()

if __name__ == "__main__":
    main()