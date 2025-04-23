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

def centering_torch(M: torch.Tensor) -> torch.Tensor:
    """
    One‑sided centering: return M @ H
    where H = I - 1/n · 11^T
    """
    n = M.size(0)
    unit = torch.ones((n, n), device=M.device)
    I    = torch.eye(n,      device=M.device)
    H    = I - unit / n
    return M @ H


def gaussian_grammat_torch(x: torch.Tensor, sigma: torch.Tensor = None) -> torch.Tensor:
    """
    Exactly like your NumPy version:
      - Compute pairwise sq. dist.
      - Median‑heuristic for sigma with epsilon fallback.
      - Build K = exp( -0.5 * dist / sigma^2 ), in‑place.
    """
    # ensure 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # xxT and squared‐distances
    xxT   = x @ x.t()                           # [n,n]
    xnorm = torch.diag(xxT).unsqueeze(1) - xxT
    xnorm = xnorm + xnorm.t()

    # median heuristic
    if sigma is None:
        # non‑zero entries
        nz = xnorm[xnorm != 0]
        if nz.numel() > 0:
            mdist = nz.median()
            sigma = (0.5 * mdist).sqrt()
        else:
            sigma = torch.tensor(1.0, device=x.device)

    # epsilon fix if sigma==0
    # 7/3 - 4/3 - 1 is machine eps in double precision
    if sigma.item() == 0:
        eps = 7./3 - 4./3 - 1
        sigma = sigma + eps

    # build kernel matrix in‑place
    K = -0.5 * xnorm / (sigma * sigma)
    return K.exp_()


def HSIC_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Match your NumPy HSIC: trace( centering(Kx) @ centering(Ky) ) / n^2
    """
    Kx = gaussian_grammat_torch(x)
    Ky = gaussian_grammat_torch(y)
    Cx = centering_torch(Kx)
    Cy = centering_torch(Ky)
    n  = x.size(0)
    # returns a Python float
    return (Cx @ Cy).trace().item() / (n * n)

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
                 disable_drift_reset: bool = False,
                 stride: int = 3):
        """
        disable_drift_reset=True will skip the shrink/reset logic and only record HSIC values
        """

        self.total_seen = 0

        self.stride = stride


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
        n = len(self.Z)
        Z_np = np.vstack(self.Z)                            # [n, d]
        t_np = np.arange(n, dtype=float)         # [n]
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
        self.total_seen += 1
        self.Z.append(z)
        if len(self.Z) > self.max_window_size:
            self.Z.pop(0)

        if len(self.Z) >= self.min_window_size \
            and (self.total_seen % self.stride == 0):
            hsic = self._test_for_independence()
        else:
            hsic = 0.0

        self.hsic_history.append(hsic)

        # optionally skip drift reset logic
        if not self.disable_drift_reset and hsic >= self.hsic_threshold:
                    # keep shrinking until we’re below threshold or at min size
                while hsic >= self.hsic_threshold and len(self.Z) > self.min_n_items:
                    self.Z.pop(0)
                    hsic = self._test_for_independence()

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