import argparse
import os
import yaml
import pandas as pd
import torch
import tqdm

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
        file_txt=params['val_txt'],
        img_folder=params['val_imgs'],
        starting_epoch=-1,
        num_workers=16,
        shuffle = False
    )

    # checkpoint path
    ckpt = '/ceph/project/DAKI4-thermal-2025/P4/runs/ae_complex_full_1/100'
    ckpt = '/home/nieb/Projects/DAKI Projects/P4/DAWIDD/ae_complex'

    # sampling rate
    sr = 3985 # average samples pr day

    # build detectors with stride and reasonable windows
    detectors = {
        name: DAWIDD_HSIC(
            ckpt_path=ckpt,
            device=args.device,
            max_window_size=w,
            min_window_size=int(0.8 * w)
        )
        for name, w in {
            'daily':    1 * sr}.items()
    }

    # grab encoder once
    # all detectors share the same encoder
    encoder = next(iter(detectors.values())).encoder

    # processing loop
    total = len(val_loader.dataset)
    print(f"Processing {total} samples with stride={args.stride}…")
    for images, *_ in tqdm.tqdm(val_loader, total=len(val_loader), desc="Batches"):
        # images: [B, C, H, W]
        images = images.to(args.device).float()
        with torch.no_grad():
            z_batch = encoder(images).view(len(images), -1).cpu().numpy()

        # feed each latent to every detector
        for det in detectors.values():
            det.add_batch(z_batch)

    # write out CSVs
    for name, det in detectors.items():
        # unpack the history into two lists
        hsic_vals, p_vals = zip(*det.hsic_history)  # det.hsic_history is now List[Tuple[float, float]]
        df = pd.DataFrame({
            'hsic_val': hsic_vals,
            'p_value': p_vals
        })
        out_file = f"{name}_hsic_pvalues.csv"
        df.to_csv(out_file, index_label='sample_index')
        print(f"Wrote {out_file} ({len(df)} samples)")

if __name__ == '__main__':
    main()