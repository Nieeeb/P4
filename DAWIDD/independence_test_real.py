from DAWIDD_HSIC_TEST import DAWIDD_HSIC
from utils.dataloader import prepare_loader
import argparse
import os
from utils import util
import yaml
import pandas as pd
import tqdm


#Loading args from CLI
parser = argparse.ArgumentParser()
parser.add_argument('--args_file', default='utils/args.yaml', type=str)
parser.add_argument('--world_size', default=1, type=int)

args = parser.parse_args()

# args for DDP
args.local_rank = int(os.getenv('LOCAL_RANK', 0))
print(f"Local rank: {args.local_rank}")
print(f"World size: {args.world_size}")

# Setting random seed for reproducability
# Seed is 0
util.setup_seed()

#Loading config
with open(args.args_file) as cf_file:
    params = yaml.safe_load( cf_file.read())


sampling_rate_per_day = 3985

samples_all_time = 749195
ckpt = '/ceph/project/DAKI4-thermal-2025/P4/runs/ae_complex_full_1/100'
detectors = {
    'daily': DAWIDD_HSIC(ckpt, device='cuda', max_window_size=1*sampling_rate_per_day,
                         min_window_size=int(0.8*1*sampling_rate_per_day), disable_drift_reset=True),
    'weekly': DAWIDD_HSIC(ckpt, device='cuda', max_window_size=7*sampling_rate_per_day,
                          min_window_size=int(0.8*7*sampling_rate_per_day), disable_drift_reset=True),
    'monthly': DAWIDD_HSIC(ckpt, device='cuda', max_window_size=30*sampling_rate_per_day,
                           min_window_size=int(0.8*30*sampling_rate_per_day), disable_drift_reset=True),
    'all_time': DAWIDD_HSIC(ckpt, device='cuda', max_window_size=1*samples_all_time,
                             min_window_size=int(0.8*samples_all_time), disable_drift_reset=True)
}

starting_epoch = -1

validation_loader, validation_sampler = prepare_loader(args, params,
                            file_txt=params.get('val_txt'),
                            img_folder=params.get('val_imgs'),
                            starting_epoch=starting_epoch,
                            num_workers=16
                            )



idx = 0
for batch in tqdm(validation_loader, desc="Validation batches"):
    images = batch[0]
    for img in images:
        for det in detectors.values():
            det.set_input(img)
        idx += 1

# ---- Write CSV files ----
for name, det in detectors.items():
    df = pd.DataFrame({'hsic_val': det.hsic_history})
    out_fname = f"{name}_hsic_values.csv"
    df.to_csv(out_fname, index_label='sample_index')
    print(f"Wrote {out_fname} with {len(det.hsic_history)} entries")