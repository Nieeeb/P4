from utils.dataset import Dataset
import warnings
from torch.utils import data

warnings.filterwarnings("ignore")

def prepare_loader(args, params, file_txt, img_folder, starting_epoch=-1, num_workers=16, shuffle=True, chrono_difference=False):
        #Dataloading train 
        filenames = []
        
        with open(file_txt) as reader:
            for filename in reader.readlines():
                filename = filename.rstrip().split('/')[-1]
                filenames.append(img_folder + filename)

        if args.local_rank == 0:
            print(f"Number of files found for {file_txt}: {len(filenames)}")

        dataset = Dataset(filenames, params.get('input_size'), params, augment=params.get('augment'), chrono_difference=chrono_difference)
        
        if args.world_size <= 1:
            sampler = None
        else:
            sampler = data.DistributedSampler(dataset, num_replicas=args.world_size, rank=args.local_rank, shuffle=shuffle, drop_last=False)
            sampler.set_epoch(starting_epoch)
        
        loader = data.DataLoader(dataset, params.get('batch_size'), sampler=sampler,
                                num_workers=num_workers, pin_memory=True, collate_fn=Dataset.collate_fn, drop_last=False)
        
        return loader, sampler