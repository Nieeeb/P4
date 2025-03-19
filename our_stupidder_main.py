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
import sys
from datetime import timedelta

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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    #os.environ['CUDA_VISIBLE_DEVICES'] = os.getenv('LOCAL_RANK', 0)
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(hours=1))

def cleanup():
    torch.distributed.destroy_process_group()    

def train(rank, args, params):
    try:
        args.local_rank = rank
        setup(rank, args.world_size)
        #Loading model
        checkpoint_path = params.get('checkpoint_path')
        starting_epoch = 0
        if check_checkpoint(checkpoint_path):
            model, optimizer, scheduler, starting_epoch = load_latest_checkpoint(checkpoint_path)
            model.to(args.local_rank)
            print(f"Checkpoint found, starting from epoch {starting_epoch + 1}")
        else:
            print("No checkpoint found, starting new training")
            starting_epoch = 0
            model = yolo_v8_m(len(params.get('names')))
            model = model.to(args.local_rank)
            if args.world_size > 1:
                # DDP mode
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = torch.nn.parallel.DistributedDataParallel(module=model, device_ids=[args.local_rank])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, last_epoch=-1)

        if starting_epoch + 1 >= params.get('epochs'):
            print(f"Already trained for {params.get('epochs')} epochs. Exiting")
            exit
        
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
            train_sampler = data.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.local_rank, shuffle=True, drop_last=False)
            train_sampler.set_epoch(starting_epoch)

        train_loader = data.DataLoader(train_dataset, params.get('batch_size'), sampler=train_sampler,
                                num_workers=32, pin_memory=True, collate_fn=Dataset.collate_fn, drop_last=False)


        #if args.local_rank == 0:
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
            validation_sampler = data.DistributedSampler(validation_dataset, num_replicas=args.world_size, rank=args.local_rank, shuffle=True, drop_last=False)
            validation_sampler.set_epoch(starting_epoch)

        validation_loader = data.DataLoader(validation_dataset, params.get('batch_size'), sampler=validation_sampler,
                                num_workers=32, pin_memory=True, collate_fn=Dataset.collate_fn, drop_last=False)
        #validation_loader = data.DataLoader(validation_dataset, params.get('batch_size'), sampler=None,
        #                        num_workers=32, pin_memory=True, collate_fn=Dataset.collate_fn, drop_last=False)
        
        #if args.world_size > 1:
        #        # DDP mode
        #        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        #        model = torch.nn.parallel.DistributedDataParallel(module=model,
        #                                                        device_ids=[args.local_rank]
        #                                                       ) #output_device=args.local_rank
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
        
        torch.distributed.barrier()
        for epoch in range(starting_epoch, params.get('epochs')):
            m_loss = util.AverageMeter()

            if args.world_size > 1:
                train_sampler.set_epoch(epoch)
                validation_sampler.set_epoch(epoch)
                
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
                    wandb.log({
                        "Training mloss average": m_loss.avg,
                        "Raw loss": loss
                    })
                
                del loss
                
                optimizer.step()

                # Log
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'  # (GB)
                    s = ('%10s' * 2 + '%10.4g') % (f'{epoch + 1}/{params.get("epochs")}', memory, m_loss.avg)
                    p_bar.set_description(s)
                    
            
        
            #Validation
            print(f"Beginning epoch validation for epoch {epoch + 1} on GPU {args.local_rank}")
            
            v_loss = util.AverageMeter()

            with torch.no_grad():
                for _, (samples, targets, _) in enumerate(validation_loader):
                    samples, targets = samples.to(args.local_rank), targets.to(args.local_rank)
                    
                    samples = samples.float() / 255

                    #print(f"Val shape: {samples.shape} and {targets.shape}")
                    
                    outputs = model(samples)
                    vloss = criterion(outputs, targets)
                    
                    torch.distributed.reduce(vloss, torch.distributed.ReduceOp.AVG)
                    v_loss.update(vloss.item(), samples.size(0))
                    
                    del outputs
                    del vloss
                    
            print(f"GPU {args.local_rank} has completed validation")
            
            if args.local_rank == 0:
                print(f"Validation complete. Val Loss is at: {v_loss.avg}")
                wandb.log({
                    'Epoch': epoch + 1,
                    'Training Epoch Loss': m_loss.avg,
                    'Validation Loss': v_loss.avg
                })
                
            del v_loss
            del m_loss           
            
            torch.distributed.barrier()
            # Step learning rate scheduler
            scheduler.step()
            
            # Saving checkpoint
            if args.local_rank == 0:
                save_checkpoint(model, optimizer, scheduler, epoch + 1, checkpoint_path, yolo_size='m')
                
        
        cleanup()
            
    except Exception as e:
        cleanup()
        print(e)
        exit





if __name__ == "__main__":
    main()