# https://www.geeksforgeeks.org/implement-convolutional-autoencoder-in-pytorch-with-cuda/
# https://github.com/axkoenig/autoencoder/blob/master/autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self, nc=1, nfe=64, nfd=64, nz=256):
        super().__init__()
        self.nc = nc
        self.nfd = nfd
        self.nfe = nfe
        self.nz = nz

        self.encoder = nn.Sequential(
            # input (nc) x 128 x 128
            nn.Conv2d(self.nc, self.nfe, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfe),
            nn.LeakyReLU(True),
            # input (nfe) x 64 x 64
            nn.Conv2d(self.nfe, self.nfe * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfe * 2),
            nn.LeakyReLU(True),
            # input (nfe*2) x 32 x 32
            nn.Conv2d(self.nfe * 2, self.nfe * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfe * 4),
            nn.LeakyReLU(True),
            # input (nfe*4) x 16 x 16
            nn.Conv2d(self.nfe * 4, self.nfe * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfe * 8),
            nn.LeakyReLU(True),
            # input (nfe*8) x 8 x 8
            nn.Conv2d(self.nfe * 8, self.nfe * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfe * 16),
            nn.LeakyReLU(True),
            # input (nfe*16) x 4 x 4
            nn.Conv2d(self.nfe * 16, self.nz, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.nz),
            nn.LeakyReLU(True)
            # output (nz) x 1 x 1
        )

        self.decoder = nn.Sequential(
            # input (nz) x 1 x 1
            nn.ConvTranspose2d(self.nz, self.nfd * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.nfd * 16),
            nn.ReLU(True),
            # input (nfd*16) x 4 x 4
            nn.ConvTranspose2d(self.nfd * 16, self.nfd * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfd * 8),
            nn.ReLU(True),
            # input (nfd*8) x 8 x 8
            nn.ConvTranspose2d(self.nfd * 8, self.nfd * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfd * 4),
            nn.ReLU(True),
            # input (nfd*4) x 16 x 16
            nn.ConvTranspose2d(self.nfd * 4, self.nfd * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfd * 2),
            nn.ReLU(True),
            # input (nfd*2) x 32 x 32
            nn.ConvTranspose2d(self.nfd * 2, self.nfd, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfd),
            nn.ReLU(True),
            # input (nfd) x 64 x 64
            nn.ConvTranspose2d(self.nfd, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # output (nc) x 128 x 128
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

class SimpleConvAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 
                            kernel_size=3, 
                            stride=2, 
                            padding=1, 
                            output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 
                            kernel_size=3, 
                            stride=2, 
                            padding=1, 
                            output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)