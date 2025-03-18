import yaml
from nets.nn import yolo_v8_m
from utils.dataset import Dataset
import argparse
import os
import warnings
import torch
from utils import util
from torch.utils import data
import tqdm
import wandb
from utils.modeltools import load_latest_checkpoint, save_checkpoint, check_checkpoint
import torch.multiprocessing as mp

warnings.filterwarnings("ignore")

def main(): 
    #Loading args from CLI
    parser = argparse.ArgumentParser()
    #parser.add_argument('--input-size', default=384, type=int)
    #parser.add_argument('--batch-size', default=96, type=int)
    #parser.add_argument('--local_rank', default=0, type=int)
    #parser.add_argument('--epochs', default=100, type=int)
    #parser.add_argument('--train', action='store_true')
    #parser.add_argument('--test', action='store_true')
    parser.add_argument('--args_file', default='utils/args.yaml', type=str)
    parser.add_argument('--world_size', default=1, type=int)

    args = parser.parse_args()

    #args for DDP
    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    print(f"Local rank: {args.local_rank}")
    #Vi kan prøve det sådan her og hvis AI-LAB ikke har world_size system variabel kan vi bare sætte default til 8
    #args.world_size = int(os.getenv('WORLD_SIZE', 1))
    #args.world_size = torch.cuda.device_count()
    print(f"World size: {args.world_size}")

    util.setup_seed()

    #Loading config
    with open(args.args_file) as cf_file:
        params = yaml.safe_load( cf_file.read())
        
    mp.spawn(train, args=(args, params), nprocs=args.world_size, join=True)
    
    if args.world_size > 1:
        "Cleans up the distributed environment"
        torch.distributed.destroy_process_group()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train(rank, args, params):
    args.local_rank = rank
    setup(rank, args.world_size)
    #Loading model
    checkpoint_path = params.get('checkpoint_path')
    starting_epoch = 0
    if check_checkpoint(checkpoint_path):
        model, optimizer, scheduler, starting_epoch = load_latest_checkpoint(checkpoint_path)
        print(f"Checkpoint found, starting from epoch {starting_epoch}")
    else:
        print("No checkpoint found, starting new training")
        starting_epoch = 0
        model = yolo_v8_m(len(params.get('names')))
        model = model.to(args.local_rank)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, last_epoch=-1)

    
    #Dataloading train 
    filenames = []
    with open(params.get('train_txt')) as reader:
        for filename in reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append(params.get('train_imgs') + filename)

    train_dataset = Dataset(filenames, params.get('input_size'), params, augment=False)


    if args.world_size <= 1:
        train_sampler = None
    else:
        train_sampler = data.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.local_rank, shuffle=True)
        train_sampler.set_epoch(starting_epoch)

    train_loader = data.DataLoader(train_dataset, params.get('batch_size'), sampler=train_sampler,
                             num_workers=16, pin_memory=True, collate_fn=Dataset.collate_fn)


 
    #Dataloading Validation
    filenames = []
    with open(params.get('val_txt')) as reader:
        for filename in reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append(params.get('val_imgs') + filename)
    
    validation_dataset = Dataset(filenames, params.get('input_size'), params, augment=False)


    if args.world_size <= 1:
        validation_sampler = None
    else:
        validation_sampler = data.DistributedSampler(validation_dataset, num_replicas=args.world_size, rank=args.local_rank, shuffle=True)
        validation_sampler.set_epoch(starting_epoch)

    validation_loader = data.DataLoader(validation_dataset, params.get('batch_size'), sampler=validation_sampler,
                             num_workers=16, pin_memory=True, collate_fn=Dataset.collate_fn)

    
    if args.world_size > 1:
            # DDP mode
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                            device_ids=[args.local_rank]
                                                            ) #output_device=args.local_rank
    criterion = util.ComputeLoss(model, params)

    num_batch = len(train_loader)

    
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
    
    for epoch in range(starting_epoch, params.get('epochs')):
        m_loss = util.AverageMeter()

        if args.world_size > 1:
            train_sampler.set_epoch(epoch)
            
        p_bar = enumerate(train_loader)
        if args.local_rank == 0:
            print(('\n' + '%10s' * 3) % ('epoch', 'memory', '    train_loss'))
        if args.local_rank == 0:    
            p_bar = tqdm.tqdm(p_bar, total=num_batch)  # progress bar

        for _, (samples, targets, _) in p_bar:
            samples, targets = samples.to(args.local_rank), targets.to(args.local_rank)
            #Model set to train
            model.train()
            
            optimizer.zero_grad()

            samples = samples.float() / 255
            #targets = targets.cuda()

            outputs = model(samples)  # forward
            loss = criterion(outputs, targets)

            m_loss.update(loss.item(), samples.size(0))


            loss *= params.get('batch_size')  # loss scaled by batch_size
            loss *= args.world_size  # gradient averaged between devices in DDP mode

            loss.backward()
            
            if args.local_rank == 0:
                wandb.log(
                    {"Training loss": m_loss.avg}
                )
            
            del loss
            
            optimizer.step()

            # Log
            if args.local_rank == 0:
                memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'  # (GB)
                s = ('%10s' * 2 + '%10.4g') % (f'{epoch + 1}/{params.get("epochs")}', memory, m_loss.avg)
                p_bar.set_description(s)
                
        
    
        #Validation
        if args.local_rank == 0:
            print(f"Beginning epoch validation for epoch {epoch}")
        #model.eval()
        #p_bar = enumerate(validation_loader)

        #if args.local_rank == 0:
        #        p_bar = tqdm.tqdm(p_bar, total=num_val_batch)  # progress bar
        
        running_vloss = 0.0

        with torch.no_grad():
            for _, (samples, targets, _) in enumerate(validation_loader):
                
                samples = samples.cuda().float() / 255
                targets = targets.cuda()

                #print(f"Val shape: {samples.shape} and {targets.shape}")
                
                outputs = model(samples)
                vloss = criterion(outputs, targets)
                running_vloss += vloss

        avg_vloss = running_vloss / (len(validation_loader.dataset) + 1)

        #print(f"Validation loss for epoch {epoch} is: {avg_vloss}")
        if args.local_rank == 0:
            wandb.log({
                'Validation Loss': avg_vloss
            })
        
        del avg_vloss
        del running_vloss
        
        # Step learning rate scheduler
        scheduler.step()
        
        if args.local_rank == 0:
            wandb.log({
                'Epoch': epoch,
                'Training Epoch Loss': m_loss.avg
            })
            
        del m_loss
        
        # Saving checkpoint
        if args.local_rank == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)





if __name__ == "__main__":
    main()