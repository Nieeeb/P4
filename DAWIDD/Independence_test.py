from DAWIDD_HSIC_TEST import DAWIDD_HSIC
from utils.dataloader import prepare_loader
import argparse
import os
from utils import util
import yaml

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




detector = DAWIDD_HSIC(
    ckpt_path = '/ceph/project/DAKI4-thermal-2025/P4/runs/ae_complex_full_1/100',
    nc=1, nfe=64, nfd=64, nz=256,
    device='cuda',
    max_window_size=90,
    min_window_size=70,
    hsic_threshold=1e-3
)


starting_epoch = -1

validation_loader, validation_sampler = prepare_loader(args, params,
                            file_txt=params.get('val_txt'),
                            img_folder=params.get('val_imgs'),
                            starting_epoch=starting_epoch,
                            num_workers=16
                            )


# stream your data (e.g. from a DataLoader)
for batch in validation_loader:     # batch shape (B, C, H, W)
    for img in batch:
        if detector.set_input(img):
            print(">>> Drift detected!")
            break
    if detector.detected_change():
        break