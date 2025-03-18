from utils.modeltools import load_latest_checkpoint, save_checkpoint, check_checkpoint
from nets.nn import yolo_v8_n
import torch
import os



def main():
    run_path = "runs/run0"
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)
    
    if check_checkpoint(run_path):
        model, optimizer, scheduler, starting_epoch = load_latest_checkpoint(run_path)
    else:
        starting_epoch = -1
        model = yolo_v8_n(num_classes=4).cuda()
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model, device_ids=[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, last_epoch=starting_epoch)
    
    for epoch in range(starting_epoch, 11):
        # Fake training
        print(epoch)
        save_checkpoint(model, optimizer, scheduler, epoch, run_path)
    
    torch.distributed.destroy_process_group()
        


if __name__ == "__main__":
    main()