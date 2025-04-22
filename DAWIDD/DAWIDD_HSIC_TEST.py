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

def centering_torch(K):
    n = K.size(0)
    I = torch.eye(n, device=K.device)
    H = I - torch.ones(n, n, device=K.device) / n
    return H @ K @ H

def gaussian_grammat_torch(x: torch.Tensor, σ=None):
    # x: [n, d] on self.device
    xx = x @ x.t()
    xnorm = xx.diag().unsqueeze(1) - xx
    xnorm = xnorm + xnorm.t()
    if σ is None:
        nz = xnorm[xnorm > 0]
        σ = (0.5 * nz.median()).sqrt() if nz.numel() else 1.0
    return torch.exp(-0.5 * xnorm / (σ**2))

def HSIC_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute HSIC between x ([n,d]) and y ([n,1]) *entirely on torch*.
    Returns a Python float.
    """
    Kx = gaussian_grammat_torch(x)
    Ky = gaussian_grammat_torch(y)
    Cx = centering_torch(Kx)
    Cy = centering_torch(Ky)
    hsic_val = (Cx @ Cy).trace() / (x.size(0) ** 2)
    return hsic_val.item()

# ---- PyTorch DAWIDD_HSIC ----




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
                 disable_drift_reset: bool = False):
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
        # 1) stack NumPy history and normalize time vector
        Z_np = np.vstack(self.Z)                            # [n, d]
        t_np = np.arange(self.n_items, dtype=float)         # [n]
        t_np = (t_np - t_np.mean()) / t_np.std()             # normalized

        # 2) convert to torch.Tensor on self.device
        Z = torch.from_numpy(Z_np).to(self.device).float()   # [n, d]
        t = torch.from_numpy(t_np).to(self.device).float()   # [n]
        t = t.unsqueeze(1)                                   # [n,1]

        # 3) call standalone HSIC function
        return HSIC_torch(Z, t)

    def add_batch(self, Z_batch: np.ndarray):
        for z in Z_batch:
            self._append_and_maybe_test(z)

    def _append_and_maybe_test(self, z):
        self.Z.append(z); self.n_items += 1
        if self.n_items > self.max_window_size:
            self.Z.pop(0); self.n_items -= 1

        if self.n_items >= self.min_window_size and (self.n_items % self.stride == 0):
            hsic = self._test_for_independence()
        else:
            hsic = 0.0

        self.hsic_history.append(hsic)

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