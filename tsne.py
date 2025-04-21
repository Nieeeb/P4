import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from PIL import Image
import cv2
import torch

p1 = "/home/nieb/Projects/DAKI Projects/P4/Data/images/valid/20200514_clip_21_2239_image_0000.jpg"
p2 = "/home/nieb/Projects/DAKI Projects/P4/Data/images/valid/20210307_clip_35_1849_image_0101.jpg"

im1 = cv2.imread(p1, cv2.IMREAD_UNCHANGED)
im2 = cv2.imread(p2, cv2.IMREAD_UNCHANGED)
d1 = {'im': im1}
d2 = {'im': im2}

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data / 255.0
y = mnist.target

#tsne = TSNE(n_components=2, perplexity=1, n_iter=1000, random_state=42)
#X_tsne = tsne.fit_transform(data)

