import torch
import numpy as np
from torch import nn

import sys, os

# compute P4 directory (one level up from DAWIDD/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# now this will work:
from nets.autoencoder import ConvAutoencoder

# ---- HSIC machinery (unchanged) ----

def centering(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ M @ H

def gaussian_grammat(x: np.ndarray, sigma: float = None) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    xx = x @ x.T
    xnorm = np.diag(xx) - xx + (np.diag(xx) - xx).T
    if sigma is None:
        nonzero = xnorm[xnorm != 0]
        sigma = np.sqrt(0.5 * np.median(nonzero)) if len(nonzero) > 0 else 1.0
    sigma = sigma or np.finfo(float).eps
    return np.exp(-0.5 * xnorm / (sigma**2))

def HSIC(x: np.ndarray, y: np.ndarray) -> float:
    Kx = gaussian_grammat(x)
    Ky = gaussian_grammat(y)
    return np.trace(centering(Kx) @ centering(Ky)) / (x.shape[0]**2)

# ---- PyTorch DAWIDD_HSIC ----

#/runs/ae_complex_full_1/100


class DAWIDD_HSIC:
    def __init__(self,
                 ckpt_path: str,
                 nc: int = 1,
                 nfe: int = 64,
                 nfd: int = 64,
                 nz: int = 256,
                 device: str = 'cuda',
                 max_window_size: int = 90,
                 min_window_size: int = 70,
                 hsic_threshold: float = 1e-3,
                 disable_drift_reset: bool = True):
        """
        disable_drift_reset=True will skip the shrink/reset logic and only record HSIC values
        """
        self.device = torch.device(device)
        # build and load your AE
        self.model = ConvAutoencoder(nc=nc, nfe=nfe, nfd=nfd, nz=nz).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        raw = ckpt.get('model', ckpt)
        stripped = {k.replace('module.', ''): v for k, v in raw.items()}
        self.model.load_state_dict(stripped)
        self.model.eval()

        self.encoder = self.model.encoder

        # sliding-window params
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.min_n_items = int(min_window_size / 4)
        self.hsic_threshold = hsic_threshold
        self.disable_drift_reset = disable_drift_reset

        # buffers
        self.Z = []              # list of latent codes
        self.n_items = 0
        self.hsic_history = []   # record all HSIC values

    def _test_for_independence(self) -> float:
        Z = np.vstack(self.Z)
        t = np.arange(self.n_items, dtype=float)
        t = (t - t.mean()) / t.std()
        return HSIC(Z, t.reshape(-1, 1))

    def add_batch(self, z: np.ndarray):
        """Add one latent code and record HSIC value."""
        self.Z.append(z)
        self.n_items += 1

        # enforce max window
        if self.n_items > self.max_window_size:
            self.Z.pop(0)
            self.n_items -= 1

        # compute HSIC if we have enough samples; else zero
        if self.n_items >= self.min_window_size:
            hsic_val = self._test_for_independence()
        else:
            hsic_val = 0.0

        # record HSIC every step
        self.hsic_history.append(hsic_val)

        # optionally skip drift reset logic
        if not self.disable_drift_reset:
            if hsic_val >= self.hsic_threshold:
                # shrink window until below threshold or at min size
                while hsic_val >= self.hsic_threshold and self.n_items > self.min_n_items:
                    self.Z.pop(0)
                    self.n_items -= 1
                    hsic_val = self._test_for_independence()

    def set_input(self, img) -> None:
        """Process one image/sample and update HSIC history."""
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.ndim == 3:
            img = img.unsqueeze(0)
        img = img.to(self.device).float()

        with torch.no_grad():
            z = self.encoder(img)
        z = z.view(z.size(0), -1).cpu().numpy()[0]

        self.add_batch(z)