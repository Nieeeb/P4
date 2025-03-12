from utils.modeltools import load_latest_checkpoint, save_checkpoint, check_checkpoint
from nets.nn import yolo_v8_n
import torch



def main():
    run_path = "runs/run0"
    
    if check_checkpoint(run_path):
        model, optimizer, scheduler, starting_epoch = load_latest_checkpoint(run_path)
    else:
        starting_epoch = -1
        model = yolo_v8_n(num_classes=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, last_epoch=starting_epoch)
    
    for epoch in range(starting_epoch, 11):
        # Fake training
        print(epoch)
        save_checkpoint(model, optimizer, scheduler, epoch, run_path)
        


if __name__ == "__main__":
    main()