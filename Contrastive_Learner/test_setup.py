import torch
from contrastive_learner import ContrastiveLearner
from nets.autoencoder import ConvAutoencoder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image



def load_encoder(checkpoint_path: str = r'/ceph/project/DAKI4-thermal-2025/P4/runs/ae_complex_full_2/50', 
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


def load_contrastive_learner():
    backbone = load_encoder() # Modify checkpoint path
    learner = ContrastiveLearner(net=backbone,
                                 image_size=128,
                                 hidden_layer=-1, # Get outputs of the final layer
                                 project_dim=128, # producerer x-længde vector
                                 use_nt_xent_loss=True,
                                 temperature = 0.1,     
                                 augment_both = True        
                                 )
    return learner



class Dataset(Dataset):
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
        return self.transform(Image.open(self.paths[idx]).convert('L'))

    
def train_learner(epochs: int = 2):
    learner = load_contrastive_learner()
    optimzer = torch.optim.Adam(learner.parameters(), lr=3e-4)
    dataset = Dataset()
    data_loader = DataLoader(dataset, batch_size=60, shuffle=True)
    for epoch in range(epochs):
        for images in data_loader:
            optimzer.zero_grad()
            loss = learner(images)
            loss.backward()
            optimzer.step()
