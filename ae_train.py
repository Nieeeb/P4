import yaml
import argparse
import os
import warnings
import torch
from utils import util
import wandb
from utils.modeltools import save_checkpoint, load_or_create_state
import torch.multiprocessing as mp
from datetime import timedelta
from utils.dataloader import prepare_loader
from train import setup, cleanup, train_epoch, validate_epoch

warnings.filterwarnings("ignore")

def main(): 
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
    
    # Creating training instances for each GPU
    mp.spawn(train, args=(args, params), nprocs=args.world_size, join=True)  

def train(rank, args, params):
    try:
        # Defining world size and creating/connecting to DPP instance
        args.local_rank = rank
        setup(rank, args.world_size)
        
        # Loading model
        # Loads if a valid checkpoint is found, otherwise creates a new model
        model, optimizer, scheduler, starting_epoch = load_or_create_state(args, params)

        if starting_epoch + 1 >= params.get('epochs'):
            print(f"Already trained for {params.get('epochs')} epochs. Exiting")
            exit
        
        #Dataloading train
        train_loader, train_sampler = prepare_loader(args, params,
                                    file_txt=params.get('train_txt'),
                                    img_folder=params.get('train_imgs'),
                                    starting_epoch=starting_epoch
                                    )

        #Dataloading Validation
        validation_loader, validation_sampler = prepare_loader(args, params,
                                    file_txt=params.get('val_txt'),
                                    img_folder=params.get('val_imgs'),
                                    starting_epoch=starting_epoch
                                    )
        
        # Defining loss function for training
        criterion = torch.nn.MSELoss()
        
        # Init Wandb
        if args.local_rank == 0:
            wandb.init(
                project="Thermal",
                config=params,
                resume="allow",
                group=params.get('run_name'),
                id=params.get('run_name')
            )
        
        if args.local_rank == 0:
            wandb.log({
                'Args File': args.args_file
            })
        
        # Pauses all worker threads to sync up GPUs before training
        torch.distributed.barrier()
        
        # Begin training
        if args.local_rank == 0:
            print("Beginning training...")
        for epoch in range(starting_epoch, params.get('epochs')):
            if args.local_rank == 0:
                print(f"Traning for epoch {epoch + 1}")
            m_loss = train_epoch(args, params,
                        model = model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        train_loader=train_loader,
                        train_sampler=train_sampler,
                        criterion=criterion,
                        epoch=epoch,
                        model_type=params.get("model_type")
                        )
            if args.local_rank == 0:
                print(f"Validation for epoch {epoch + 1}")
            v_loss = validate_epoch(args, params,
                                model = model,
                                validation_loader=validation_loader,
                                validation_sampler=validation_sampler,
                                criterion=criterion,
                                epoch=epoch,
                                model_type=params.get("model_type")
                                )    
        
            if args.local_rank == 0:
                print(f"Validation for epoch {epoch} complete. Val Loss is at: {v_loss.avg}")
                wandb.log({
                    'Epoch': epoch + 1,
                    'Training Epoch Loss': m_loss.avg,
                    'Validation Loss': v_loss.avg
                })
                
                del m_loss
                del v_loss
            
            # Saving checkpoint
            if args.local_rank == 0:
                save_checkpoint(model, optimizer, scheduler, epoch + 1, params.get('checkpoint_path'), yolo_size='ae')
        
        # Training complete
        if args.local_rank == 0:
                print(f"Training Completed succesfully\nTrained {params.get('epochs')} epochs")
        
        torch.distributed.barrier() # Pauses all worker threads to sync up GPUs
        cleanup() # Destroy DDP process group
            
    except Exception as e:
        cleanup()
        print(e)
        exit

if __name__ == "__main__":
    main()