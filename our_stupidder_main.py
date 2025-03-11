import yaml
from nets.nn import yolo_v8_n
from utils.dataset import Dataset
import argparse
import os
import torch
from utils import util
from torch.utils import data
import tqdm


def main(): 
    #Loading args from CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=500, type=int)
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
    with open(r'utils\args.yaml') as cf_file:
        params = yaml.safe_load( cf_file.read())




def train(args, params):

    #Loading model
    model = yolo_v8_n(len(params.get("names")))

    #Dataloading
    filenames = []
    with open('Data/train.txt') as reader:
        for filename in reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append('/home/nieb/Projects/DAKI Mini Projects/fmlops-1/Data/images/train/' + filename)

    dataset = Dataset(filenames, args.input_size, params, True, augment=False)


    if args.world_size <= 1:
        sampler = None
    else:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, sampler,
                             num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)


    if args.world_size > 1:
            # DDP mode
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                            device_ids=[args.local_rank],
                                                            output_device=args.local_rank)
    criterion = util.ComputeLoss(model, params)
    optimizer = torch.optim.Adam(model.parameters, lr=0.002)

    #Model set to train
    model.train()
    num_batch = len(loader)
    for epoch in range(args.epochs):


        if args.world_size > 1:
            sampler.set_epoch(epoch)
            p_bar = enumerate(loader)
        if args.local_rank == 0:
            print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'loss'))
        if args.local_rank == 0:
            p_bar = tqdm.tqdm(p_bar, total=num_batch)  # progress bar

        for _, (samples, targets, _) in p_bar:
            optimizer.zero_grad()

            samples = samples.cuda().float() / 255
            targets = targets.cuda()

            outputs = model(samples)  # forward
            loss = criterion(outputs, targets)

            loss *= args.batch_size  # loss scaled by batch_size
            loss *= args.world_size  # gradient averaged between devices in DDP mode

            loss.backward()

            optimizer.step()






if __name__ == "__main__":
    main()