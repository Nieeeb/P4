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

warnings.filterwarnings("ignore")

def main(): 
    #Loading args from CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=384, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    #args for DDP
    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    #Vi kan prøve det sådan her og hvis AI-LAB ikke har world_size system variabel kan vi bare sætte default til 8
    args.world_size = int(os.getenv('WORLD_SIZE', 1))

    if args.world_size > 1:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    util.setup_seed()

    #Loading config
    with open(r'utils/args.yaml') as cf_file:
        params = yaml.safe_load( cf_file.read())
        
    train(args, params)




def train(args, params):
    #Loading model
    checkpoint_path = params.get('checkpoint_path')
    starting_epoch = 0
    if check_checkpoint(checkpoint_path):
        model, optimizer, scheduler, starting_epoch = load_latest_checkpoint(checkpoint_path)
        print(f"Checkpoint found, starting from epoch {epoch}")
    else:
        print("No checkpoint found, starting new training")
        starting_epoch = 0
        model = yolo_v8_m(len(params.get('names')))
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, last_epoch=-1)

    
    #Dataloading train 
    filenames = []
    with open('Data/train.txt') as reader:
        for filename in reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append(params.get('train_imgs') + filename)

    train_dataset = Dataset(filenames, args.input_size, params, augment=False)


    if args.world_size <= 1:
        train_sampler = None
    else:
        train_sampler = data.distributed.DistributedSampler(train_dataset)

    train_loader = data.DataLoader(train_dataset, args.batch_size, train_sampler,
                             num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)


    #Dataloading Validation
    filenames = []
    with open('Data/val.txt') as reader:
        for filename in reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append(params.get('val_imgs') + filename)

    validation_dataset = Dataset(filenames, args.input_size, params, augment=False)


    if args.world_size <= 1:
        validation_sampler = None
    else:
        validation_sampler = data.distributed.DistributedSampler(validation_dataset)

    validation_loader = data.DataLoader(validation_dataset, args.batch_size, validation_sampler,
                             num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)

    



    if args.world_size > 1:
            # DDP mode
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                            device_ids=[args.local_rank],
                                                            output_device=args.local_rank)
    criterion = util.ComputeLoss(model, params)

    num_batch = len(train_loader)

    #num_val_batch = len(validation_loader)
    
    # Init Wandb
    wandb.init(
        project="Thermal",
        config=params
    )
    
    for epoch in range(starting_epoch, args.epochs):

        m_loss = util.AverageMeter()

        if args.world_size > 1:
            train_sampler.set_epoch(epoch)
            
        p_bar = enumerate(train_loader)
        if args.local_rank == 0:
            print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'train_loss'))
        if args.local_rank == 0:
            p_bar = tqdm.tqdm(p_bar, total=num_batch)  # progress bar

        for _, (samples, targets, _) in p_bar:
            #Model set to train
            model.train()
            
            optimizer.zero_grad()

            samples = samples.cuda().float() / 255
            targets = targets.cuda()

            #print(f"Train shape: {samples.shape} and {targets.shape}")

            outputs = model(samples)  # forward
            loss = criterion(outputs, targets)

            m_loss.update(loss.item(), samples.size(0))


            loss *= args.batch_size  # loss scaled by batch_size
            loss *= args.world_size  # gradient averaged between devices in DDP mode

            loss.backward()
            
            wandb.log(
                {"Training loss": m_loss.avg}
            )
            
            del loss


            optimizer.step()


            # Log
            if args.local_rank == 0:
                memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'  # (GB)
                s = ('%10s' * 2 + '%10.4g') % (f'{epoch + 1}/{args.epochs}', memory, m_loss.avg)
                p_bar.set_description(s)
            
    
        #Validation
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

        avg_vloss = running_vloss / (len(p_bar) + 1)

        #print(f"Validation loss for epoch {epoch} is: {avg_vloss}")
        wandb.log({
            'Validation Loss': avg_vloss
        })
        
        del avg_vloss
        del running_vloss
        
        # Step learning rate scheduler
        scheduler.step()
        
        # Saving checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)





if __name__ == "__main__":
    main()