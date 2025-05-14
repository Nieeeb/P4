import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import random

class DataClusterer():
    def __init__(self, K, data):
        self.kmeans = KMeans(
                    n_clusters=K,
                    random_state=0,
                    n_init=10,
                )
        self.kmeans.fit(data)
        
    def predict_cluster(self, x) -> int:
        prediction = self.kmeans.predict(x)
        return prediction
    
def elbow_test_kmeans(data, max_clusters):
    random_state = 0
    sse_list = []
    
    for k in tqdm(range(1, max_clusters), desc="Fitting Cluster Sizes", total=max_clusters):
        kmeans = KMeans(n_clusters=k,
                        random_state=random_state,
                        n_init=10
                        )
        kmeans.fit(data)
        sse = kmeans.inertia_
        sse_list.append(sse)
    
    print(sse_list)
    plt.plot(range(1, max_clusters), sse_list)
    plt.xticks(range(1, max_clusters))
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("SSE")
    plt.title("SSE versus K")
    plt.grid()
    plt.savefig("elbow_test_SSE.png")
    plt.show()

def write_cluster_txts(data: pd.DataFrame):
    txt_files = defaultdict(list)
    random.seed(0)
    
    for index, row in tqdm(data.iterrows(), desc="Grouping filenames by cluster", total=len(data)):
        txt_files[row['cluster']].append(row['filename'])
    
    for cluster, file_names in tqdm(txt_files.items(), desc="Writing .txts from clusters", total=len(txt_files)):
        path = f"Data/febtrain_ae_cluster_{cluster}.txt"
        random.shuffle(file_names)
        with open(path, 'w') as f:
            for file in file_names:
                f.write(file + "\n")

def main():               
    train_path = "DAWIDD/flatten_ae_febtrain.csv"
    data = pd.read_csv(train_path, index_col=0)
    print(data.head())
    skip_columns = ['filename', 'datetime']
    selected_columns = [col for col in data.columns if col not in skip_columns]
    x = data[selected_columns]
    
    val_path = "DAWIDD/flatten_ae_febvalid.csv"
    val_data = pd.read_csv(val_path, index_col=0)
    val_x = val_data[selected_columns]
    
    # tests looks like 10 or 11 clusters might be right
    #elbow_test_kmeans(x, 30)
    cluster_alg = DataClusterer(K=11, data=x)
    
    train_predictions = cluster_alg.predict_cluster(x)
    val_predictions = cluster_alg.predict_cluster(val_x)
    
    data['cluster'] = train_predictions
    data.to_csv("DAWIDD/flatten_ae_febtrain_with_clusters.csv")
    val_data['cluster'] = val_predictions
    val_data.to_csv("DAWIDD/flatten_ae_febval_with_clusters.csv")
    
    print(data.head())
    write_cluster_txts(data)

if __name__ == '__main__':
    main()
