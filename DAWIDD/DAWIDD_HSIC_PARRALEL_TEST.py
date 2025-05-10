import numpy as np
import torch
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

_FINFO = {
    torch.float32: torch.finfo(torch.float32).eps,
    torch.float16: torch.finfo(torch.float16).eps,
    torch.float64: torch.finfo(torch.float64).eps,
}

def _median_heuristic(dist2: torch.Tensor) -> torch.Tensor:
    nz = dist2[dist2 != 0]
    if nz.numel() == 0:
        return torch.tensor(1.0, device=dist2.device, dtype=dist2.dtype)
    return (0.5 * nz.median()).sqrt()


def gaussian_grammat_torch(
    x: torch.Tensor,
    sigma=None
) -> torch.Tensor:
    if x.ndim == 2:
        x = x.unsqueeze(0)
        single = True
    elif x.ndim == 3:
        single = False
    else:
        raise ValueError("x must be 2-D or 3-D")

    B, n, _ = x.shape
    xxT   = x @ x.transpose(1, 2)
    diag  = torch.diagonal(xxT, dim1=1, dim2=2)
    dist2 = diag.unsqueeze(2) - 2*xxT + diag.unsqueeze(1)

    if sigma is None:
        sigmas = []
        for b in range(B):
            s = _median_heuristic(dist2[b])
            s = torch.clamp(s, min=_FINFO[x.dtype])
            sigmas.append(s)
        sigma = torch.stack(sigmas)
    else:
        sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
        if sigma.ndim == 0:
            sigma = sigma.repeat(B)
        elif sigma.ndim == 1 and sigma.numel() == B:
            pass
        else:
            raise ValueError("Bad shape for sigma")
        sigma = torch.clamp(sigma, min=_FINFO[x.dtype])

    denom = (2.0 * sigma.pow(2)).view(B, 1, 1)
    K = torch.exp(-dist2 / denom)
    return K[0] if single else K


def centering_torch(K: torch.Tensor) -> torch.Tensor:
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


def HSIC_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.inference_mode():
        Kx = centering_torch(gaussian_grammat_torch(x))
        Ky = centering_torch(gaussian_grammat_torch(y))
        n  = x.size(0)
        return (Kx @ Ky).trace().item() / (n * n)


def HSIC_batch(Xb: torch.Tensor, Yb: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        Kx = centering_torch(gaussian_grammat_torch(Xb))
        Ky = centering_torch(gaussian_grammat_torch(Yb))
        n  = Xb.size(1)
        return torch.einsum('bii->b', Kx @ Ky) / (n * n)


class DAWIDD_HSIC:
    def __init__(
        self,
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
        perm_batch_size: int = 100
    ):
        self.total_seen = 0
        self.stride = stride
        if device == 'cuda':
            device = 'cuda:0'
        self.device = torch.device(device)

        
        #ckpt = 'Data/temp/latest'
        ckpt = '/ceph/project/DAKI4-thermal-2025/P4/runs/ae_complex_full_2/50'
        model = ConvAutoencoder(nc=1, nfe=64, nfd=64, nz=256).to(device)
        ckpt = torch.load(ckpt, map_location=device)
        raw = ckpt.get('model', ckpt)
        stripped = {k.replace('module.', ''): v for k, v in raw.items()}
        model.load_state_dict(stripped)
        model.eval()
        self.encoder = model.encoder

        # sliding-window params
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.min_n_items = int(min_window_size / 4)
        self.hsic_threshold = hsic_threshold
        self.disable_drift_reset = disable_drift_reset

        self.Z = []
        self.n_items = 0
        self.hsic_history = []

        # permutation testing params (single GPU)
        self.perm_reps = perm_reps
        self.perm_batch_size = perm_batch_size
        self.perm_device = self.device

    def _hsic_pvalue_batch(self, x, y):
        T_obs = HSIC_torch(x, y)
        n, d = x.size(0), x.size(1)
        null_stats = []

        Xb_cpu = x.view(1, n, d).cpu()
        y_cpu = y.cpu()

        for start in range(0, self.perm_reps, self.perm_batch_size):
            b = min(self.perm_batch_size, self.perm_reps - start)
            perm_dev = self.perm_device

            Xb = Xb_cpu[:b].to(perm_dev)
            y_chunk = y_cpu.to(perm_dev)
            perms = torch.stack([torch.randperm(n, device=perm_dev) for _ in range(b)])
            Yb = y_chunk[perms]

            Knull = HSIC_batch(Xb, Yb)
            null_stats.append(Knull.cpu())

            del Xb, Yb, Knull, perms

        null = torch.cat(null_stats)
        p_val = (null.ge(T_obs).sum().item() + 1) / (len(null) + 1)
        return T_obs, p_val

    def _test_for_independence_perm_fast(self):
        n = len(self.Z)
        Z_np = np.vstack(self.Z)
        t_np = np.arange(n, dtype=float)
        t_np = (t_np - t_np.mean()) / t_np.std()

        Z = torch.from_numpy(Z_np).to(self.device).float()
        t = torch.from_numpy(t_np).to(self.device).float().unsqueeze(1)

        return self._hsic_pvalue_batch(Z, t)

    def add_batch(self, Z_batch: np.ndarray):
        for z in Z_batch:
            self._append_and_maybe_test(z)

    def _append_and_maybe_test(self, z):
        self.total_seen += 1
        self.Z.append(z)
        if len(self.Z) > self.max_window_size:
            self.Z.pop(0)
        if len(self.Z) >= self.min_window_size and (self.total_seen % self.stride == 0):
            T, p = self._test_for_independence_perm_fast()
        else:
            T, p = 0.0, 1.0
        self.hsic_history.append((T, p))

    def set_input(self, img) -> None:
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.ndim == 3:
            img = img.unsqueeze(0)
        img = img.to(self.device).float()

        with torch.no_grad():
            z = self.encoder(img)
        z = z.view(z.size(0), -1).cpu().numpy()[0]
        self.add_batch(z)
