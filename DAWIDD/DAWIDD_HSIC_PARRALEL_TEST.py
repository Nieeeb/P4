import torch
import numpy as np
from torch import nn
from torch.cuda.amp import autocast
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
    Batched one-sided centering: return M @ H
    where H = I - 1/n · 11^T.
    Supports both M.shape = [n, n] and [B, n, n].
    """
    # Lift 2D → 3D
    if M.dim() == 2:
        M = M.unsqueeze(0)  # [1, n, n]
        single = True
    elif M.dim() == 3:
        single = False
    else:
        raise ValueError(f"Expected M to be 2D or 3D, got {M.dim()}D")

    B, n, _ = M.shape
    unit = torch.ones((n, n), device=M.device, dtype=M.dtype)
    I    = torch.eye(n,      device=M.device, dtype=M.dtype)
    H    = I - unit / n       # [n, n]

    # batch-matrix multiply: [B, n, n] @ [n, n] → [B, n, n]
    out = M @ H

    return out[0] if single else out


def gaussian_grammat_torch(x: torch.Tensor, sigma: torch.Tensor = None) -> torch.Tensor:
    """
    Batched Gaussian RBF kernel:
      - x.shape = [n, d]   → returns [n, n]
      - x.shape = [B, n, d] → returns [B, n, n]
    Median-heuristic for sigma applied per batch if sigma=None.
    """
    # Lift 2D → 3D
    if x.dim() == 2:
        x = x.unsqueeze(0)  # [1, n, d]
        single = True
    elif x.dim() == 3:
        single = False
    else:
        raise ValueError(f"Expected x to be 2D or 3D, got {x.dim()}D")

    B, n, d = x.shape
    # compute pairwise inner products and squared distances
    xxT   = x @ x.transpose(1, 2)                           # [B, n, n]
    diag  = torch.diagonal(xxT, dim1=1, dim2=2).unsqueeze(2)  # [B, n, 1]
    xnorm = diag - xxT
    xnorm = xnorm + xnorm.transpose(1, 2)                   # [B, n, n]

    # compute sigma per batch if needed
    if sigma is None:
        sigmas = []
        for b in range(B):
            nz = xnorm[b][xnorm[b] != 0]
            if nz.numel() > 0:
                mdist = nz.median()
                s = (0.5 * mdist).sqrt()
            else:
                s = torch.tensor(1.0, device=x.device, dtype=x.dtype)
            # fallback if zero
            if s.item() == 0:
                eps = (7./3 - 4./3 - 1)
                s = s + eps
            sigmas.append(s)
        sigma = torch.stack(sigmas)  # [B]
    else:
        sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
        if sigma.dim() == 0:
            sigma = sigma.repeat(B)
        elif sigma.dim() == 1 and sigma.numel() == B:
            pass
        else:
            raise ValueError(f"Invalid sigma shape {sigma.shape} for batch size {B}")

    # build kernel
    denom = (sigma**2).view(B, 1, 1)                         # [B, 1, 1]
    K = torch.exp(-0.5 * xnorm / denom)                     # [B, n, n]

    return K[0] if single else K

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







def HSIC_batch(Xb, Yb):
    # Xb: [B, n, d], Yb: [B, n, 1]
    Kx = centering_torch(gaussian_grammat_torch(Xb))
    Ky = centering_torch(gaussian_grammat_torch(Yb))
    M = Kx @ Ky            # [B, n, n]
    # sum over the diagonal for each batch
    traces = torch.einsum('bii->b', M)      # or: M.diagonal(dim1=1, dim2=2).sum(1)
    return traces.float() / (Xb.size(1)**2)


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
                 stride: int = 10,
                 perm_reps: int = 200,
                 perm_batch_size: int = 100,
                 perm_devices: list = None):
        """
        disable_drift_reset=True will skip the shrink/reset logic and only record HSIC values
        """

        self.total_seen = 0

        self.stride = stride

        if device == 'cuda':
            device = 'cuda:0'
        self.device = torch.device(device)

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


        self.perm_reps        = perm_reps
        self.perm_batch_size  = perm_batch_size
        # default: use every cuda device except the one holding self.device
        if perm_devices is None:
            all_ids = list(range(torch.cuda.device_count()))
            main_id = self.device.index if self.device.type=='cuda' else None
            self.perm_devices = [i for i in all_ids if i!=main_id]
        else:
            self.perm_devices = perm_devices
        if not self.perm_devices:
            # fallback to main device
            self.perm_devices = [ self.device.index or 0 ]


    def _hsic_pvalue_batch(self, x, y):
        T_obs = HSIC_torch(x, y)
        n, d = x.size(0), x.size(1)
        null_stats = []

        # prepare the “clean” Xb on CPU once
        Xb_cpu = x.view(1,n,d).cpu()  # [1,n,d] on CPU
        y_cpu = y.cpu()

        for chunk_idx, start in enumerate(range(0, self.perm_reps, self.perm_batch_size)):
            b = min(self.perm_batch_size, self.perm_reps - start)
            # pick a GPU
            dev_id = self.perm_devices[chunk_idx % len(self.perm_devices)]
            perm_dev = torch.device(f'cuda:{dev_id}')

            # move chunked Xb and generate perms on that GPU
            Xb = Xb_cpu[:b].to(perm_dev)    
            y_chunk = y_cpu.to(perm_dev)            # [b,n,d] on gpu:dev_id
            perms = torch.stack([torch.randperm(n, device=perm_dev)
                                for _ in range(b)])
            Yb = y_chunk[perms]                   # [b,n,1]

            Knull = HSIC_batch(Xb, Yb)                  # runs on gpu:dev_id
            null_stats.append(Knull.cpu())              # move back to CPU

            # free gpu:dev_id
            del Xb, Yb, Knull, perms
            torch.cuda.empty_cache()

        null = torch.cat(null_stats)                   # on CPU
        p    = (null.ge(T_obs).sum().item() + 1) / (len(null) + 1)
        return T_obs, p


    def _test_for_independence_perm_fast(self, B=1000, batch_size=100):
        # stack and normalize
        n = len(self.Z)
        Z_np = np.vstack(self.Z)                  # [n, d]
        t_np = np.arange(n, dtype=float)
        t_np = (t_np - t_np.mean()) / t_np.std()

        # to tensors once
        Z = torch.from_numpy(Z_np).to(self.device).float()   # [n, d]
        t = torch.from_numpy(t_np).to(self.device).float()   # [n]
        t = t.unsqueeze(1)                                   # [n, 1]

    # batch‐permutation p‐value returns exactly 2 values
        T_obs, p_val = self._hsic_pvalue_batch(Z, t)
        return T_obs, p_val


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
            T, p = self._test_for_independence_perm_fast(B=200)
        else:
            T, p = 0.0, 1.0
        self.hsic_history.append((T, p))




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