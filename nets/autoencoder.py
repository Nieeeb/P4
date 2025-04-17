# https://www.geeksforgeeks.org/implement-deep-autoencoder-in-pytorch-for-image-reconstruction/
import torch

# Creating a DeepAutoencoder class 
class DeepAutoencoder(torch.nn.Module): 
    def __init__(self): 
        super().__init__()         
        self.encoder = torch.nn.Sequential( 
            torch.nn.Linear(28 * 28, 256), 
            torch.nn.ReLU(), 
            torch.nn.Linear(256, 128), 
            torch.nn.ReLU(), 
            torch.nn.Linear(128, 64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(64, 10) 
        ) 
        
        self.decoder = torch.nn.Sequential( 
            torch.nn.Linear(10, 64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(64, 128), 
            torch.nn.ReLU(), 
            torch.nn.Linear(128, 256), 
            torch.nn.ReLU(), 
            torch.nn.Linear(256, 28 * 28), 
            torch.nn.Sigmoid() 
        ) 

    def forward(self, x): 
        encoded = self.encoder(x) 
        decoded = self.decoder(encoded) 
        return decoded