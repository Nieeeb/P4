from utils.modeltools import load_saved_cluster_models, load_single_model
from utils.util import setup_seed, non_max_suppression, scale, box_iou, compute_ap
import warnings
import torch
from tqdm import tqdm
import numpy
import pandas as pd
import argparse
import os
import yaml
from utils.dataloader import prepare_loader
import torchvision
from sklearn.decomposition import PCA
from nets.autoencoder import ConvAutoencoder
from sklearn.cluster import KMeans
import time
from contrastive_learner.contrastive_learner import ContrastiveLearner
import random


warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_file', default='utils/testing_args.yaml', type=str)
    parser.add_argument('--world_size', default=1, type=int)

    args = parser.parse_args()

    # args for DDP
    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    print(f"Local rank: {args.local_rank}")
    print(f"World size: {args.world_size}")

    # Setting random seed for reproducability
    # Seed is 0
    setup_seed()

    #Loading config
    with open(args.args_file) as cf_file:
        params = yaml.safe_load( cf_file.read())
    
    params['world_size'] = args.world_size
    params['local_rank'] = args.local_rank
    
    test_across_months(args, params)

class ContrastiveEncoder(torch.nn.Module):
    def __init__(self, learner: ContrastiveLearner):
        super().__init__()
        
        self.encoder = learner.net
        self.projection = learner.projection
        
    def flatten(self, t):
        return t.reshape(t.shape[0], -1)
        
    def forward(self, x):
        x = self.encoder(x)
        if torch.isinf(x).any():
            #print("triggered")
            finite_mask = torch.isfinite(x)
            
            mean_val = x[finite_mask].mean()
            # Replace both +inf and -inf
            inf_mask = ~finite_mask & ~torch.isnan(x)
            
            x[inf_mask] = mean_val
            
        x = self.flatten(x)
        x = self.projection(x)

        return x

class DataClusterer():
    def __init__(self, K, data):
        random_seed = 0
        self.kmeans = KMeans(
                    n_clusters=K,
                    random_state=random_seed,
                    n_init=10,
                )
        self.kmeans.fit(data)
        
        random.seed(random_seed)
        
    def predict_cluster(self, x):
        prediction = self.kmeans.predict(x)
        return prediction
    
    def random_cluster(self, x):
        prediction = [random.randint(0, 6) for _ in range(len(x))]
        return prediction

class ModelSelecter():
    def __init__(self, args) -> None:
        self.args = args

        embedding_path = self.args['embedding_path']
        
        self.pca, self.kmeans = self.prepare_kmeans(
                                            embedding_path=embedding_path,
                                            K = 7,
                                            n_components = 2
                                            )
        
        ckpt = self.args['checkpoint_path']
        if self.args['single_model'] == True:
            self.single_model = True
            self.models = load_single_model(ckpt)
        else:
            self.single_model = False
            self.models = load_saved_cluster_models(ckpt)
        
        for model in self.models:
            model.half()
            model.cuda()
            model.eval()
        
        self.resize = torchvision.transforms.Resize((128,128))
        
        self.random = False
        
        encoder_ckpt = self.args['encoder_ckpt']
        if self.args['encoder_type'] == 'ae':
            self.encoder = self.load_ae_encoder(encoder_ckpt)
        elif self.args['encoder_type'] == 'mlp':
            self.encoder = self.load_con_encoder(encoder_ckpt)
        elif self.args['encoder_type'] == 'random':
            print("Utilizing Random Clustering")
            self.random = True
        else:
            print("Unknown encoder type")
            exit
    
    def prepare_kmeans(self, embedding_path, K=7, n_components=2):
        print("Preparing kmeans")
        data = pd.read_csv(embedding_path, index_col=0)
        skip_columns = ['filename', 'datetime', 'cluster']
        selected_columns = [col for col in data.columns if col not in skip_columns]
        x = data[selected_columns]
        
        if self.args['do_pca'] == True:
            pca = PCA(n_components=n_components, random_state=0)
            x = pca.fit_transform(x)
        else:
            pca = None
        
        kmeans = DataClusterer(K, x)
        
        return pca, kmeans
    
    def load_ae_encoder(self, path):
        print("Loading Autoencoder")
        ckpt = torch.load(path)
        state_dict = ckpt['model']
        
        model = ConvAutoencoder()
        model.load_state_dict(state_dict=state_dict)
        encoder = model.encoder
        encoder.half()
        encoder.cuda()
        encoder.eval()
        return encoder
    
    def load_con_encoder(self, path):
        print("Loading Contrastive Encoder")
        ckpt = torch.load(path)
        state_dict = ckpt['learner']
        
        conv = ConvAutoencoder()
        backbone = conv.encoder
        learner = ContrastiveLearner(net = backbone,
                                image_size= 128,
                                hidden_layer=-1,
                                augment_both=True,
                                use_nt_xent_loss=True,
                                project_dim=256
                                )
        learner.load_state_dict(state_dict)
        
        encoder = ContrastiveEncoder(learner)
        encoder.half()
        encoder.cuda()
        encoder.eval()
        
        return encoder
    
    def predict_clusters(self, samples):
        with torch.no_grad():
            resized_samples = self.resize(samples)
            outputs = self.encoder(resized_samples)

            encodings = []
            
            for output in outputs:
                flat = output.cpu().numpy().flatten().tolist()
                encodings.append(flat)
                
            x = pd.DataFrame(encodings, columns=[x for x in range(len(encodings[0]))])
            if self.pca is not None:
                x = self.pca.transform(x)

            clusters = self.kmeans.predict_cluster(x)
            return clusters
        
    def make_predictions(self, samples):
        with torch.no_grad():
            if self.random == True and self.single_model == False:
                clusters = self.kmeans.random_cluster(samples)
            elif self.single_model == False:
                clusters = self.predict_clusters(samples)
            elif self.single_model == True:
                predictions = self.models[0](samples)
                return predictions

            predictions = []
            for idx, sample in enumerate(samples):
                # Addint batch dimension
                sample = sample.unsqueeze(0)
                model_id = clusters[idx]
                # Indexing into model list from clusters. Assuming cluster x corresponds to model at index x
                prediction = self.models[model_id](sample)
                predictions.append(prediction)

            tensor = torch.cat(predictions)
            return tensor

def test_across_months(params, args):
    
    selecter = ModelSelecter(args)
    
    test_txts = [
        "Data/test_January.txt",
        "Data/test_February.txt",
        "Data/test_March.txt",
        "Data/test_April.txt",
        "Data/test_May.txt",
        "Data/test_June.txt",
        "Data/test_July.txt",
        "Data/test_August.txt"
    ]
    
    #Dataloading Validation
    results = []
    for txt in test_txts:
        filenames = []
        
        with open(txt) as reader:
            for filename in reader.readlines():
                filename = filename.rstrip().split('/')[-1]
                filenames.append(args['val_imgs'] + filename)
        
        args['val_txt'] = txt
        cache_path_override = f"Data/images/{txt.split('/')[-1]}.cache"
        loader, _ = prepare_loader(args=params,
                                params=args,
                                file_txt=txt,
                                img_folder=args['val_imgs'],
                                cache_path_override=cache_path_override,
                                num_workers=32
                                )

        map50, mean_ap, test_duration, per_sample_time = test(modelselector=selecter,
                            loader=loader)
        result = {
            'txt': txt,
            'map50': map50,
            'mean_ap': mean_ap,
            'num_images': len(filenames),
            'test_duration': test_duration,
            'per_sample_time': per_sample_time,
            'encoder': args['encoder_type'],
            'encodings_file': args['embedding_path'],
        }
        print(result)
        results.append(result)
    df = pd.DataFrame(results)
    df.to_csv(f"{args['result_path']}")
    print("--------- Testing Complete -----------")

def test(modelselector: ModelSelecter, loader):
    with torch.no_grad():
        # Configure
        iou_v = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
        n_iou = iou_v.numel()

        m_pre = 0.
        m_rec = 0.
        map50 = 0.
        mean_ap = 0.
        metrics = []
        p_bar = tqdm(enumerate(loader), desc=('%10s' * 3) % ('precision', 'recall', 'mAP'), total=len(loader))

        clock_start = False
        for idx, (samples, targets, shapes) in p_bar:
            if clock_start == False:
                start_time = time.time()
                clock_start = True
            samples = samples.cuda()
            targets = targets.cuda()
            samples = samples.half()  # uint8 to fp16/32
            samples = samples / 255  # 0 - 255 to 0.0 - 1.0
            _, _, height, width = samples.shape  # batch size, channels, height, width

            outputs = modelselector.make_predictions(samples)
            # Inference
            #outputs = model(samples)

            # NMS
            targets[:, 2:] *= torch.tensor((width, height, width, height)).cuda()  # to pixels
            outputs = non_max_suppression(outputs, 0.001, 0.65)

            # Metrics
            for i, output in enumerate(outputs):
                labels = targets[targets[:, 0] == i, 1:]
                correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

                if output.shape[0] == 0:
                    if labels.shape[0]:
                        metrics.append((correct, *torch.zeros((3, 0)).cuda()))
                    continue

                detections = output.clone()
                scale(detections[:, :4], samples[i].shape[1:], shapes[i][0], shapes[i][1])

                # Evaluate
                if labels.shape[0]:
                    tbox = labels[:, 1:5].clone()  # target boxes
                    tbox[:, 0] = labels[:, 1] - labels[:, 3] / 2  # top left x
                    tbox[:, 1] = labels[:, 2] - labels[:, 4] / 2  # top left y
                    tbox[:, 2] = labels[:, 1] + labels[:, 3] / 2  # bottom right x
                    tbox[:, 3] = labels[:, 2] + labels[:, 4] / 2  # bottom right y
                    scale(tbox, samples[i].shape[1:], shapes[i][0], shapes[i][1])

                    correct = numpy.zeros((detections.shape[0], iou_v.shape[0]))
                    correct = correct.astype(bool)

                    t_tensor = torch.cat((labels[:, 0:1], tbox), 1)
                    iou = box_iou(t_tensor[:, 1:], detections[:, :4])
                    correct_class = t_tensor[:, 0:1] == detections[:, 5]
                    for j in range(len(iou_v)):
                        x = torch.where((iou >= iou_v[j]) & correct_class)
                        if x[0].shape[0]:
                            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                            matches = matches.cpu().numpy()
                            if x[0].shape[0] > 1:
                                matches = matches[matches[:, 2].argsort()[::-1]]
                                matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                                matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
                            correct[matches[:, 1].astype(int), j] = True
                    correct = torch.tensor(correct, dtype=torch.bool, device=iou_v.device)
                metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))
            
        end_time = time.time()
        test_duration = end_time - start_time
        per_sample_time = test_duration / len(loader.dataset)
        
        # Compute metrics
        metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to numpy
        if len(metrics) and metrics[0].any():
            tp, fp, m_pre, m_rec, map50, mean_ap = compute_ap(*metrics)

        # Print results
        print('%10.3g' * 3 % (m_pre, m_rec, mean_ap))

    return map50, mean_ap, test_duration, per_sample_time

if __name__ == "__main__":
    main()