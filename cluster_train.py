import yaml
import argparse
import os
import warnings
import torch
from utils import util
import wandb
from utils.modeltools import save_checkpoint, load_current_cluster, load_or_create_cluster
import torch.multiprocessing as mp
from datetime import timedelta
from utils.dataloader import prepare_loader
import torchvision

warnings.filterwarnings("ignore")

def main(): 
    #Loading args from CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_file', default='utils/cluster_args.yaml', type=str)
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
def train_epoch(args, params, model, optimizer, scheduler, train_loader, train_sampler, criterion, epoch, model_type="yolo", resize=False):
    m_loss = util.AverageMeter()

    # If in DDP, sampler needs current epoch
    # Used to determine which data shuffle to use if GPUs get desynced
    if args.world_size > 1:
        train_sampler.set_epoch(epoch)
        
    # Model set to train
    model.train()

    # Iterates through the training set
    for batchidx, (samples, targets, shapes) in enumerate(train_loader):
        # Sends data to appropriate GPU device
        samples, targets = samples.to(args.local_rank), targets.to(args.local_rank)
        
        if resize:
            resize = torchvision.transforms.Resize((128,128))
            samples = resize(samples)
        
        optimizer.zero_grad()

        samples = samples.float() / 255 # Input images are 8 bit single channel images. Converts to 0-1 floats

        outputs = model(samples)  # forward pass
        if model_type == "yolo":
            loss = criterion(outputs, targets) # Calculate training loss
        elif model_type == "deep_ae":
            loss = criterion(outputs, samples)

        m_loss.update(loss.item(), samples.size(0))

        loss *= params.get('batch_size')  # loss scaled by batch_size
        loss *= args.world_size  # gradient averaged between devices in DDP mode

        loss.backward() # Backpropagation

        # Logging to wandb
        if args.local_rank == 0:
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

# Function that validates the model on a validation set
# Intended to be used during training and not for testing performance of model
def validate_epoch(args, params, model, validation_loader, validation_sampler, criterion, epoch, model_type="yolo", resize=False):
    print(f"Beginning epoch validation for epoch {epoch + 1} on GPU {args.local_rank}")     
    v_loss = util.AverageMeter()
    
    # If in DDP, sampler needs current epoch
    # Used to determine which data shuffle to use if GPUs get desynced
    if args.world_size > 1:
        validation_sampler.set_epoch(epoch)

    # Iterates through validation set
    # Disables gradient calculations
    with torch.no_grad():
        for batchidx, (samples, targets, shapes) in enumerate(validation_loader):
            # Sending data to appropriate GPU
            samples, targets = samples.to(args.local_rank), targets.to(args.local_rank)
            
            if resize:
                resize = torchvision.transforms.Resize((128,128))
                samples = resize(samples)
            
            samples = samples.float() / 255 # Input images are 8 bit single channel images. Converts to 0-1 floats
            
            outputs = model(samples) # Forward pass
            
            if model_type == "yolo":
                vloss = criterion(outputs, targets) # Calculate loss
            elif model_type == "deep_ae":
                vloss = criterion(outputs, samples)
            
            torch.distributed.reduce(vloss, torch.distributed.ReduceOp.AVG) # Syncs loss and takes the average across GPUs
            v_loss.update(vloss.item(), samples.size(0))
            
            del outputs
            del vloss
            
    print(f"GPU {args.local_rank} has completed validation")
        
    return v_loss

def train(rank, args, params):
    try:
        # Defining world size and creating/connecting to DPP instance
        args.local_rank = rank
        setup(rank, args.world_size)
        
        # Her bliver cluster skabt. Denne function giver dig et dictionarymed følgende format:
        states = {
        'clusters': ['LISTE AF DICTIONARYIES MED MODEL STATES'],
        'current_cluster': 0 #Int med index for hvilken model, aka model 2 hører til cluster 2
        }
        
        # Model states ser således ud. selfølgelig ikke strings, men de rigtige datatyper
        cluster = {
        'model': 'yolo model',
        'optimizer': 'adam optimizer',
        'scheduler': 'steplrscheduler',
        'epoch': 0
        }
        
        states = load_or_create_cluster(args, params)
        
        # Vi har også en txt fil til hvert cluster. Det er filerne train_ae_cluster_x.txt, hvor x er cluster index

        if starting_epoch + 1 >= params.get('epochs'):
            print(f"Already trained for {params.get('epochs')} epochs. Exiting")
            exit
        
        #Dataloading train
        train_loader, train_sampler = prepare_loader(args, params,
                                    file_txt=params.get('train_txt'),
                                    img_folder=params.get('train_imgs'),
                                    starting_epoch=starting_epoch,
                                    num_workers=16
                                    )

        #Dataloading Validation
        validation_loader, validation_sampler = prepare_loader(args, params,
                                    file_txt=params.get('val_txt'),
                                    img_folder=params.get('val_imgs'),
                                    starting_epoch=starting_epoch,
                                    num_workers=16
                                    )
        
        # Defining loss function for training
        criterion = util.ComputeLoss(model, params)
        
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
                save_checkpoint(model, optimizer, scheduler, epoch + 1, params.get('checkpoint_path'), yolo_size='m')
        
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