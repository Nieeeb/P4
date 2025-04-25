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

import torch
from typing import Tuple, Union

_FINFO = {
    torch.float32: torch.finfo(torch.float32).eps,
    torch.float16: torch.finfo(torch.float16).eps,
    torch.float64: torch.finfo(torch.float64).eps,
}

def _median_heuristic(dist2: torch.Tensor) -> torch.Tensor:
    """
    Median heuristic for σ – identical to the NumPy reference:

        ‖x_i-x_j‖² = 2 σ²  ⇒  σ = sqrt( 0.5 * median(dist²_ij) )
    """
    # ignore the diagonal zeros
    nz = dist2[dist2 != 0]
    if nz.numel() == 0:
        return torch.tensor(1.0, device=dist2.device, dtype=dist2.dtype)
    return (0.5 * nz.median()).sqrt()


# ──────────────────────────────────────────────────────────────────────────────
def gaussian_grammat_torch(
    x: torch.Tensor,
    sigma: Union[None, torch.Tensor, float] = None
) -> torch.Tensor:
    """
    Gaussian RBF Gram matrix – behaviour matches the reference implementation
    1:1, but is fully vectorised and GPU friendly.
        • Accepts x with shape  [n, d]         → returns [n, n]
                                  or [B, n, d] → returns [B, n, n]
        • If `sigma` is None  → median heuristic per *batch*
        • If `sigma` is a scalar tensor / float → same σ for every batch
        • If `sigma` is a 1-D tensor of length B → σ[b] for batch b
    """
    # lift to 3-D so we can treat both cases uniformly
    if x.ndim == 2:
        x = x.unsqueeze(0)              # [1, n, d]
        single = True
    elif x.ndim == 3:
        single = False
    else:
        raise ValueError("x must be 2-D or 3-D")

    B, n, _ = x.shape
    # ‖x_i-x_j‖²  =  ‖x_i‖² + ‖x_j‖² − 2 x_i·x_j
    xxT   = x @ x.transpose(1, 2)                       # [B, n, n]
    diag  = torch.diagonal(xxT, dim1=1, dim2=2)         # [B, n]
    dist2 = diag.unsqueeze(2) - 2*xxT + diag.unsqueeze(1)  # [B, n, n]

    # ── bandwidth σ ──────────────────────────────────────────────────────────
    if sigma is None:
        sigmas = []
        for b in range(B):
            s = _median_heuristic(dist2[b])
            # NumPy version adds machine-epsilon if σ == 0
            s = torch.clamp(s, min=_FINFO[x.dtype])
            sigmas.append(s)
        sigma = torch.stack(sigmas)                     # [B]
    else:
        sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
        if sigma.ndim == 0:                 # scalar
            sigma = sigma.repeat(B)
        elif sigma.ndim == 1 and sigma.numel() == B:    # one per batch
            pass
        else:
            raise ValueError("Bad shape for sigma")

        sigma = torch.clamp(sigma, min=_FINFO[x.dtype]) # make strictly > 0

    # ── Gaussian kernel K_ij = exp(-‖x_i-x_j‖² / 2σ²) ───────────────────────
    denom = (2.0 * sigma.pow(2)).view(B, 1, 1)          # [B, 1, 1]
    K = torch.exp(-dist2 / denom)                       # [B, n, n]

    return K[0] if single else K


def centering_torch(K: torch.Tensor) -> torch.Tensor:
    """
    One-sided centering  (identical to NumPy’s  K @ H).
    Works for [n, n] and [B, n, n].
    """
    if K.ndim == 2:
        K = K.unsqueeze(0)
        single = True
    elif K.ndim == 3:
        single = False
    else:
        raise ValueError("K must be 2-D or 3-D")

    B, n, _ = K.shape
    H = torch.eye(n, device=K.device, dtype=K.dtype) - 1.0 / n
    Kc = K @ H
    return Kc[0] if single else Kc
# ---------- scalar & batched HSIC (GPU, no autograd, float32) -----------------

def HSIC_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    d = 2 case, NumPy-compatible scalar HSIC.
    All math stays in float32 on the chosen device (no autocast / FP16),
    so the value is bit-for-bit identical to the CPU reference as long as the
    GPU follows IEEE-754 (CUDA does).
    """
    with torch.inference_mode():
        Kx = centering_torch(gaussian_grammat_torch(x))
        Ky = centering_torch(gaussian_grammat_torch(y))
        n  = x.size(0)
        hsic = (Kx @ Ky).trace() / (n * n)
    return hsic.item()


def HSIC_batch(Xb: torch.Tensor, Yb: torch.Tensor) -> torch.Tensor:
    """
    Same estimator but for a *batch* of permutations:
        Xb: [B, n, d]   (same X repeated B times)
        Yb: [B, n, 1]   (different permutations of y)
    Returns: [B]  (float32, on same device as the inputs)
    """
    with torch.inference_mode():
        Kx = centering_torch(gaussian_grammat_torch(Xb))    # [B, n, n]
        Ky = centering_torch(gaussian_grammat_torch(Yb))    # [B, n, n]
        n  = Xb.size(1)
        traces = torch.einsum('bii->b', Kx @ Ky) / (n * n)   # [B]
    return traces




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
            

        null = torch.cat(null_stats)                   # on CPU
        p    = (null.ge(T_obs).sum().item() + 1) / (len(null) + 1)
        return T_obs, p


    def _test_for_independence_perm_fast(self):
        n = len(self.Z)
        Z_np = np.vstack(self.Z)                  # [n, d]
        t_np = np.arange(n, dtype=float)
        t_np = (t_np - t_np.mean()) / t_np.std()

        Z = torch.from_numpy(Z_np).to(self.device).float()   # [n, d]
        t = torch.from_numpy(t_np).to(self.device).float()   # [n]
        t = t.unsqueeze(1)                                   # [n, 1]

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
            T, p = self._test_for_independence_perm_fast()
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