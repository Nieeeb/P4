import yaml
import argparse
import os
import warnings
import torch
from utils import util
import wandb
from utils.modeltools import save_checkpoint_dasr, load_or_create_dasr
import torch.multiprocessing as mp
from datetime import timedelta
from MoCo.Dataset import GrayscalePatchDataset
from torch.utils.data import DataLoader, DistributedSampler

warnings.filterwarnings("ignore")

def main(): 
    #Loading args from CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_file', default='utils/dasr_args.yaml', type=str)
    parser.add_argument('--world_size', default=1, type=int)

    args = parser.parse_args()

    # args for DDP
    args.local_rank = int(os.getenv('LOCAL_RANK', 0))

    # Setting random seed for reproducability
    # Seed is 0
    util.setup_seed()

    #Loading config
    with open(args.args_file) as cf_file:
        params = yaml.safe_load( cf_file.read())
        
    params['local_rank'] = args.local_rank
    params['world_size'] = args.world_size
    params['args_file'] = args.args_file
    
    print(f"Local rank: {params['local_rank']}")
    print(f"World size: {params['world_size']}")
    
    # Creating training instances for each GPU
    #mp.spawn(train, args=(args, params), nprocs=args.world_size, join=True)
    train(args.local_rank, args, params)

# Function for defining machine and port to use for DDP
# Sets up the process group
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    #os.environ['CUDA_VISIBLE_DEVICES'] = os.getenv('LOCAL_RANK', 0)
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(hours=1))

# Method to gracefully destory the process group
# If this is not done, problems may arise on future runs
def cleanup():
    torch.distributed.destroy_process_group()    

# Function for training an entire epoch
# Logs to wandb
def train_epoch(args, moco_model, optimizer, scheduler, train_loader, train_sampler, loss_fn, epoch):
    m_loss = util.AverageMeter()

    # If in DDP, sampler needs current epoch
    # Used to determine which data shuffle to use if GPUs get desynced
    #if args['world_size'] > 1:
    #    train_sampler.set_epoch(epoch)
        
    # Model set to train
    moco_model.train()

    # Iterates through the training set
    for batchidx, (q, k, _) in enumerate(train_loader):
        # Sends data to appropriate GPU device
        q, k = q.to(args['local_rank']), k.to(args['local_rank'])
        
        optimizer.zero_grad()

        embeddings, logits, labels = moco_model(q, k)
            
        loss = loss_fn(logits, labels)
        
        m_loss.update(loss.item(), q.size(0))
        
        loss.backward()
        
        # Logging to wandb
        if args['local_rank'] == 0:
            e = epoch * len(train_loader) + 1
            s = batchidx
            step = e + s
            wandb.log({
                "Training step": step,
                "Training mloss average": m_loss.avg,
                "Raw loss": loss
            })

        del loss # Deletes loss to save memory
        
        optimizer.step() # Steps the optimizer

    scheduler.step() # Step learning rate scheduler
    
    return m_loss

def train(rank, params, args):
    try:
        # Defining world size and creating/connecting to DPP instance
        args['local_rank'] = rank
        setup(rank, args['world_size'])
        
        # Loading model
        # Loads if a valid checkpoint is found, otherwise creates a new model
        moco_model, optimizer, scheduler, starting_epoch = load_or_create_dasr(args)

        if starting_epoch + 1 >= args['epochs']:
            print(f"Already trained for {args['epochs']} epochs. Exiting")
            exit
        
        # Dataloader
        dataset = GrayscalePatchDataset(patch_size=args['patch_size'],
                                        files_txt=args['train_txt'],
                                        img_folder=args['train_imgs'])
        #train_sampler = DistributedSampler(dataset=dataset, shuffle=True, drop_last=True)
        #train_sampler.set_epoch(starting_epoch)
        #train_loader = DataLoader(dataset, batch_size=args['batch_size'], sampler=train_sampler, drop_last=True, pin_memory=True, num_workers=args['num_workers'])
        train_loader = DataLoader(dataset, batch_size=args['batch_size'], drop_last=True, pin_memory=True, num_workers=args['num_workers'])
        
        loss_fn = torch.nn.CrossEntropyLoss().to(args['local_rank'])
        
        # Init Wandb
        if args['local_rank'] == 0:
            wandb.init(
                project="Thermal",
                config=args,
                resume="allow",
                group=args['run_name'],
                id=args['run_name']
            )
        
        if args['local_rank'] == 0:
            wandb.log({
                'Args File': args['args_file']
            })
        
        # Pauses all worker threads to sync up GPUs before training
        print(f"GPU {args['local_rank']} is ready")
        #torch.distributed.barrier()
        
        # Begin training
        if args['local_rank'] == 0:
            print("Beginning training...")
            
        for epoch in range(starting_epoch, args['epochs']):
            if args['local_rank'] == 0:
                print(f"Traning for epoch {epoch + 1}")
                
            m_loss = train_epoch(args,
                        moco_model = moco_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        train_loader=train_loader,
                        train_sampler=None,
                        loss_fn=loss_fn,
                        epoch=epoch
                        )  
        
            if args['local_rank'] == 0:
                print(f"Training for epoch {epoch + 1} complete. Train Loss is at: {m_loss.avg}")
                wandb.log({
                    'Epoch': epoch + 1,
                    'Training Epoch Loss': m_loss.avg,
                })
                
                del m_loss
            
            # Saving checkpoint
            if args['local_rank'] == 0:
                save_checkpoint_dasr(moco_model, optimizer, scheduler, epoch + 1, args['checkpoint_path'])
        
        # Training complete
        if args['local_rank'] == 0:
                print(f"Training Completed succesfully\nTrained {args['epochs']} epochs")
        
        #torch.distributed.barrier() # Pauses all worker threads to sync up GPUs
        #cleanup() # Destroy DDP process group
            
    except Exception as e:
        #cleanup()
        print(e)
        exit

if __name__ == "__main__":
    main()