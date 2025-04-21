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
import cv2
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

def display_boxes(samples, targets, shapes):
    colors = [
        (255, 0 , 0),
        (128, 128, 0),
        (0, 0, 255),
        (0, 255, 0)
    ]
    names = [
        'person',
        'bike',
        'motorcycle',
        'vehicle'
    ]
    
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    
    images = []
    for i, sample in enumerate(samples):
        matches = []
        for target in targets:
            if i == int(target[0].item()):
                matches.append(target)
        
        image = None
        boxes = []
        labels = targets[targets[:, 0] == i, 1:]

        if labels.shape[0]:
            tbox = labels[:, 1:5].clone()  # target boxes
            tbox[:, 0] = labels[:, 1] - labels[:, 3] / 2  # top left x
            tbox[:, 1] = labels[:, 2] - labels[:, 4] / 2  # top left y
            tbox[:, 2] = labels[:, 1] + labels[:, 3] / 2  # bottom right x
            tbox[:, 3] = labels[:, 2] + labels[:, 4] / 2  # bottom right y
            
            #util.scale(tbox, samples[i].shape[1:], shapes[i][0], shapes[i][1])
            util.scale(tbox, samples[i].shape[1:], shapes[i][0], None)
            
            image = np.transpose(sample.cpu().numpy(), (1, 2, 0))
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            for bi, box in enumerate(tbox):
                category = int(matches[bi][1].item())
                cv2.rectangle(image, (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item())), colors[category], 1)
                cv2.putText(image, names[category], (int(box[0].item()), int(box[1].item())), font, 1, colors[category], 1)
                item = (int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item()))
                boxes.append(item)
            images.append(image)
            #for box in enumerate(tbox):
            #    print(box[0])
            #    item = [box[0], box[1], box[2], box[3]]
            #    boxes.append(item)
            #    print(item)
    return images, boxes

def display_targets(samples, targets, shapes):
    _, _, height, width = samples.shape  # batch size, channels, height, width
    targets[:, 2:] *= torch.tensor((width, height, width, height)).cuda()  # to pixels
    images, boxes = display_boxes(samples, targets, shapes)
    return images, boxes

# Does not work yet
def display_outputs(samples, outputs, shapes):
    # NMS
    outputs = util.non_max_suppression(outputs, 0.001, 0.65)
    detections = None
    images = []
    for i, output in enumerate(outputs):
        detections = output.clone()
        detections = util.scale(detections[:, :4], samples[i].shape[1:], shapes[i][0], shapes[i][1])
        images.append(display_boxes(samples, detections, shapes))
    return images

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
    
    # Loading model
    # Loads if a valid checkpoint is found, otherwise creates a new model
    model, optimizer, scheduler, starting_epoch = load_or_create_state(args, params)
    
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
                                    starting_epoch=-1
                                    )
    
    # Iterates through validation set
    # Disables gradient calculations
    with torch.no_grad():
        model.eval()
        for data_idx, (samples, targets, shapes) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # Sending data to appropriate GPU
            samples, targets = samples.to(args.local_rank), targets.to(args.local_rank)
            
            samples = samples / 255
            # Inference
            outputs = model(samples)
            for output in outputs:
                for row in output[1]:
                    print(row)
            
            #rows = outputs.shape[1]
            
            boxes = []
            scores = []
            class_ids = []
            CLASSES = [
                    'person',
                    'bike',
                    'motorcycle',
                    'vehicle'
                    ]
            
            
            # Iterate through output to collect bounding boxes, confidence scores, and class IDs
            for i in range(rows):
                classes_scores = outputs[1][i][4:]
                print(classes_scores)
                (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
                if maxScore >= 0.25:
                    box = [
                        outputs[0][i][0] - (0.5 * outputs[0][i][2]),  # x center - width/2 = left x
                        outputs[0][i][1] - (0.5 * outputs[0][i][3]),  # y center - height/2 = top y
                        outputs[0][i][2],  # width
                        outputs[0][i][3],  # height
                    ]
                    boxes.append(box)
                    scores.append(maxScore)
                    class_ids.append(maxClassIndex)

            # Apply NMS (Non-maximum suppression)
            result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

            detections = []

            # Iterate through NMS results to draw bounding boxes and labels
            for i in range(len(result_boxes)):
                index = result_boxes[i]
                box = boxes[index]
                detection = {
                    "class_id": class_ids[index],
                    "class_name": CLASSES[class_ids[index]],
                    "confidence": scores[index],
                    "box": box,
                }
                detections.append(detection)
                print(detection)
            
            #images, boxes = display_targets(samples, targets, shapes)
            #print(boxes)
            # Does not work yet
            #images, boxes = display_outputs(samples, outputs, shapes)
            
            #for i, image in enumerate(images):
            #    cv2.imshow(f"{i}", image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            break
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()