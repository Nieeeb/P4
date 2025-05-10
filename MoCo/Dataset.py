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
from Moco import MoCo
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
        self.transform = transforms.Compose([transforms.ToTensor()]) #gør billeder til tensor

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        W, H = img.size
        ps = self.patch_size

        def rand_crop():
            x = random.randint(0, W-ps)
            y = random.randint(0, H-ps)
            return self.transform(img.crop((x, y, x+ps, y+ps)))
        x1, x2 = rand_crop(), rand_crop()
        return x1, x2, self.paths[idx]

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

def loss_function(q, k, queue):
    τ = 0.05

    # N is the batch size
    N = q.shape[0]
    
    # C is the dimensionality of the representations
    C = q.shape[1]

    # bmm stands for batch matrix multiplication
    # If mat1 is a b×n×m tensor, mat2 is a b×m×p tensor, 
    # then output will be a b×n×p tensor. 
    pos = torch.exp(torch.div(torch.bmm(q.view(N,1,C), k.view(N,C,1)).view(N, 1),τ))
    
    # performs matrix multiplication between query and queue tensors
    neg = torch.sum(torch.exp(torch.div(torch.mm(q.view(N,C), torch.t(queue)),τ)), dim=1)
    
    # sum is over positive as well as negative samples
    denominator = neg + pos

    return torch.mean(-torch.log(torch.div(pos,denominator)))

def create_queue(dataloader, device, model_k, K, batch_size):
    queue = None
    flag = 0
    if queue is None:
        while True:

            with torch.no_grad():
                for q, k, _ in tqdm(dataloader, desc="Filling  Queue", total=K/batch_size):            

                    k = k.to(device)
                    k = model_k(k)
                    k = k.detach()

                    k = torch.div(k,torch.norm(k,dim=1).reshape(-1,1))

                    if queue is None:
                        queue = k
                    else:
                        if queue.shape[0] < K:
                            queue = torch.cat((queue, k), 0)    
                        else:
                            flag = 1
                    
                    if flag == 1:
                        return queue

            if flag == 1:
                return queue

def main():
    args = load_args()
    setup_seed()
    setup(0, 1)
    dataset = GrayscalePatchDataset(patch_size=args['patch_size'])
    sampler = DistributedSampler(dataset=dataset, shuffle=True, drop_last=False)
    sampler.set_epoch(-1)
    loader = DataLoader(dataset, batch_size=args['batch_size'], sampler=sampler, drop_last=False, pin_memory=True)
    # plot_patch_pairs(loader)

    moco_model = MoCo(Encoder,
                dim=256, K=args['queue_size'],
                m=args['momentum'], T=args['temperature'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    moco_model = moco_model.to(device)

    optimizer = torch.optim.Adam(
        moco_model.module.encoder_q.parameters() if isinstance(moco_model, nn.DataParallel) 
        else moco_model.encoder_q.parameters(),
        lr=args['lr'], weight_decay=1e-4
    )

    num_epochs = 1
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    scaler = torch.amp.GradScaler(device=device)
    
    # defining our deep learning architecture
    model_q = resnet18(pretrained=False)
    model_q.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(model_q.fc.in_features, 100)),
        ('added_relu1', nn.ReLU(inplace=True)),
        ('fc2', nn.Linear(100, 50)),
        ('added_relu2', nn.ReLU(inplace=True)),
        ('fc3', nn.Linear(50, 25))
    ]))

    model_q.fc = classifier
    
    
    
    #model_q = Encoder()
    model_k = copy.deepcopy(model_q)
    
    model_q.to(device)
    model_k.to(device)
    
    queue = create_queue(
        dataloader=loader,
        device=device,
        model_k=model_k,
        K = args['queue_size'],
        batch_size=args['batch_size']
    )

    for epoch in range(num_epochs):
        moco_model.train()
        sampler.set_epoch(epoch)
        for q, k, _ in loader:
            #for i in range(10000):
            optimizer.zero_grad()
            q = q.to(device)
            k = k.to(device)
            #embeddings, logits, labels = moco_model(im_q, im_k)
            #print(logits)
            #print(moco_model.queue.shape)

            q = model_q(q)
            k = model_k(k)
            k = k.detach()
            
            # normalize the ouptuts, make them unit vectors
            q = torch.div(q,torch.norm(q,dim=1).reshape(-1,1))
            k = torch.div(k,torch.norm(k,dim=1).reshape(-1,1))
            
            loss = loss_function(q, k, queue)
            
            loss.backward()
            
            #loss = loss_fn(logits, labels)

            #scaled_loss = scaler.scale(loss)
            #loss.backward()
            #scaled_loss.backward()
            optimizer.step()
            #scaler.step(optimizer)
            #scaler.update()
            
            # update the queue, adding the batch to it
            queue = torch.cat((queue, k), 0) 

            # dequeue if the queue gets larger than the max queue size - denoted by K
            if queue.shape[0] > args['queue_size']:
                queue = queue[args['batch_size']:,:]
            
            # Update model K parameters with momentum
            for θ_k, θ_q in zip(model_k.parameters(), model_q.parameters()):
                θ_k.data.copy_(args['momentum']*θ_k.data + θ_q.data*(1.0 - args['momentum']))
            model_k.eval()

            print(loss)
            #break


        
        
    
    cleanup()

if __name__ == "__main__":
    main()