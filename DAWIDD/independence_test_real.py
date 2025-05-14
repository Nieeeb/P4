import argparse
import os
import yaml
import pandas as pd
import torch
import tqdm
import torch.distributed as dist
import torchvision

# import the single-GPU DAWIDD_HSIC implementation
from DAWIDD_HSIC_PARRALEL_TEST import DAWIDD_HSIC
from utils.dataloader import prepare_loader
from utils import util

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_file', type=str, default='utils/hsic_args.yaml',
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
        shuffle=False
    )
    sampler = val_loader.sampler
    local_indices = list(sampler)
    #print(local_indices)

    # checkpoint path
    ckpt = '/ceph/project/DAKI4-thermal-2025/P4/runs/ae_complex_full_2/50'
    #ckpt = 'Data/temp/50'

    # sampling rate (average clips per day)
    sr = 24 * 1
    
    # Loading saved state
    state_path = 'DAWIDD/i_am_the_hsic_state.pickle'
    
    # How often to save
    save_interval = 100
    
    if os.path.isfile(state_path):
        state = torch.load(state_path)
        print(f"Checkpoint found, starting from index {state['index']}")
    else:
        print("Checkpoint not found, creating blank state")

        # build detector (single key 'quarterly')
        detectors = {
            'Daily': DAWIDD_HSIC(
                ckpt_path=ckpt,
                device=args.device,
                max_window_size=sr,
                min_window_size=sr - 5,
                stride=9,
                perm_reps=500,
                perm_batch_size=5
            ),
            'Weekly': DAWIDD_HSIC(
                ckpt_path=ckpt,
                device=args.device,
                max_window_size=(sr * 7),
                min_window_size=(sr * 7) - 5,
                stride=9,
                perm_reps=500,
                perm_batch_size=5
            ),
            'Monthly': DAWIDD_HSIC(
                ckpt_path=ckpt,
                device=args.device,
                max_window_size=(sr * 30),
                min_window_size=(sr * 30) - 5,
                stride=9,
                perm_reps=500,
                perm_batch_size=5
            )
        }

        # grab shared encoder
        encoder = next(iter(detectors.values())).encoder
        
        state = {
            'index': 0,
            'detectors': detectors,
            'encoder': encoder
        }
    
    detectors = state['detectors']
    encoder = state['encoder']
    starting_index = state['index']

    resize = torchvision.transforms.Resize((128,128))
    
    with torch.no_grad():
        # processing loop
        for index, (images, _, _) in tqdm.tqdm(enumerate(val_loader), total=len(val_loader), desc="Batches"):
            if index < starting_index:
                continue
            else:
                images = images.to(args.device).float() / 255
                #print(images)
                images = resize(images)
                
                z_batch = encoder(images).view(len(images), -1).cpu().numpy()
                
                for det in detectors.values():
                    det.add_batch(z_batch)
                    
                if index % save_interval == 0 and index != starting_index:
                    state = {
                        'index': index,
                        'detectors': detectors,
                        'encoder': encoder
                        }
                    torch.save(state, state_path)
                    print(f"Checkpoint saved at index {index}")
                    
    torch.save(detector, "DAWIDD/complete_hsic.pickle")
    # use the 'quarterly' detector's history
    #detector = detectors['Daily', 'Ẅeekly', 'Monthly']
    for period, detector in detectors.items():
        hsic_vals, p_vals = zip(*detector.hsic_history)
    
        print(f"idx: {len(local_indices)}")
        print(f"hscic: {len(hsic_vals)}")
        print(f"pval: {len(p_vals)}")

    # build local triplets
    #local_triplets = list(zip(local_indices, hsic_vals, p_vals))

    # distributed gather
    #world_size = dist.get_world_size()
    #rank = dist.get_rank()
    #if rank == 0:
    #gather_list = [None] * world_size
    #else:
    #    gather_list = None
    #dist.gather_object(local_triplets, gather_list, dst=0)

    #if rank == 0:
        # flatten, sort, and save
        #full = [t for per_rank in gather_list for t in per_rank]
        #full.sort(key=lambda x: x[0])
    #idxs, hsics, ps = zip(*full)
        df = pd.DataFrame({
                'sample_index': local_indices,
                'hsic_val': hsic_vals,
                'p_value': p_vals
            })
        df.to_csv(f'{period}_ranks_hsic.csv', index=False)
        print(f"Wrote {period}_ranks_hsic.csv ({len(df)} rows)")


if __name__ == '__main__':
    main()
