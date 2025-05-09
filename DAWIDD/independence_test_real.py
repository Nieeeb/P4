import argparse
import os
import yaml
import pandas as pd
import torch
import tqdm

import torch.distributed as dist

from DAWIDD_HSIC_PARRALEL_TEST import DAWIDD_HSIC
from utils.dataloader import prepare_loader
from utils import util

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_file', type=str, default='utils/args.yaml',
                        help="YAML file with data paths & loader params")
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--stride', type=int, default=3,
                        help="Compute HSIC every `stride` samples")
    parser.add_argument('--device', type=str, default='cuda',
                        help="torch device for encoder & GPU HSIC")
    args = parser.parse_args()

    # DDP boilerplate (if used)
    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    print(f"Local rank: {args.local_rank}, World size: {args.world_size}")

    # reproducibility
    util.setup_seed()

    # load config
    with open(args.args_file) as f:
        params = yaml.safe_load(f)

    # data loader
    val_loader, _ = prepare_loader(
        args, params,
        file_txt=params['train_txt'],
        img_folder=params['train_imgs'],
        starting_epoch=-1,
        num_workers=16,
        shuffle = False
    )

    sampler = val_loader.sampler

    local_indices = list(sampler)  # e.g. [rank, rank+8, rank+16, …] if shuffle=False

    # checkpoint path
    ckpt = '/ceph/project/DAKI4-thermal-2025/P4/runs/ae_complex_full_1/100'
    # ckpt = '/home/nieb/Projects/DAKI Projects/P4/DAWIDD/ae_complex'

    # sampling rate
    sr = 48 * 90 # average clips pr day

    perm_gpus = list(range(1, torch.cuda.device_count()))

    # build detectors with stride and reasonable windows
    detectors = {
        name: DAWIDD_HSIC(
            ckpt_path=ckpt,
            device=args.device,
            max_window_size=sr ,
            min_window_size=int(0.8 * sr),
            stride= 120, # amount of frames pr. clip
            perm_reps=500,               
            perm_batch_size=25,
            perm_devices=perm_gpus,
        )
        for name, w in {
            'quarterly':    sr}.items()
    }

    # grab encoder once
    # all detectors share the same encoder
    encoder = next(iter(detectors.values())).encoder

    # processing loop
    total = len(val_loader.dataset)
    for images, *_ in tqdm.tqdm(val_loader, total=len(val_loader), desc="Batches"):
        # images: [B, C, H, W]
        images = images.to(args.device).float()
        with torch.no_grad():
            z_batch = encoder(images).view(len(images), -1).cpu().numpy()

        # feed each latent to every detector
        for det in detectors.values():
            det.add_batch(z_batch)


    # unpack your history
    hsic_vals, p_vals = zip(*det.hsic_history)
    
    # 2. build a list of triples (global_idx, hsic, p) on each rank
    local_triplets = list(zip(local_indices, hsic_vals, p_vals))
    
    # 3. gather them all to rank 0
    world_size = dist.get_world_size()
    rank        = dist.get_rank()
    
    if rank == 0:
        gather_list = [None] * world_size
    else:
        gather_list = None
    
    # gather_object is available in PyTorch ≥1.8
    dist.gather_object(local_triplets, gather_list, dst=0)
    
    if rank == 0:
        # flatten and sort by the original sample index
        full = [t for per_rank in gather_list for t in per_rank]
        full.sort(key=lambda x: x[0])
        
        # split back out into columns
        idxs, hsics, ps = zip(*full)
        df = pd.DataFrame({
            'sample_index': idxs,
            'hsic_val':      hsics,
            'p_value':       ps
        })
        df.to_csv('all_ranks_hsic.csv', index=False)
        print("Wrote all_ranks_hsic.csv (749000 rows)")

if __name__ == '__main__':
    main()