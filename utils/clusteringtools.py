import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import random
from kneed import KneeLocator
from sklearn.decomposition import PCA
import copy
import torch
import numpy as np


class DataClusterer():
    def __init__(self, K, data):
        random_seed = 0
        self.kmeans = KMeans(
                    n_clusters=K,
                    random_state=random_seed,
                    n_init=10,
                )
        self.kmeans.fit(data)
        
    def predict_cluster(self, x):
        prediction = self.kmeans.predict(x)
        return prediction
    
def elbow_test_kmeans(data, max_clusters):
    random_state = 0
    sse_list = []
    
    for k in tqdm(range(1, max_clusters + 1), desc="Fitting Cluster Sizes", total=max_clusters):
        kmeans = KMeans(n_clusters=k,
                        random_state=random_state,
                        n_init=10
                        )
        kmeans.fit(data)
        sse = kmeans.inertia_
        sse_list.append(sse)
        
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), sse_list)
    plt.xticks(range(1, max_clusters + 1))
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("SSE")
    plt.title("SSE versus K for 256 dim SimCLR")
    plt.grid()
    plt.savefig("elbow_test_ae_256.png")
    plt.show()
    return sse_list

def write_cluster_txts(data: pd.DataFrame):
    txt_files = defaultdict(list)
    random.seed(0)
    
    for index, row in tqdm(data.iterrows(), desc="Grouping filenames by cluster", total=len(data)):
        txt_files[row['cluster']].append(row['filename'])
    
    for cluster, file_names in tqdm(txt_files.items(), desc="Writing .txts from clusters", total=len(txt_files)):
        path = f"Data/feb_mlp_256_cluster_{cluster}_train.txt"
        random.shuffle(file_names)
        with open(path, 'w') as f:
            for file in file_names:
                f.write(file + "\n")

def super_elbow(data, max_clusters, pcas):
    sse_lists = []
    random_seed = 0
    for dimension in pcas:
        working_data = copy.deepcopy(data)
        pca = PCA(n_components=dimension,
                random_state=random_seed)
        x = pca.fit_transform(working_data)
        sse = elbow_test_kmeans(x, max_clusters)
        dct = {
            f'PCA dim {dimension}': sse
        }
        sse_lists.append(dct)
    torch.save(sse_lists, "super_elbow_ae.pickle")
    return sse_lists

def pick_k(sse_list):
    #y = sse_list[1:30]
    x = [x for x in range(len(sse_list))]
    kl = KneeLocator(x, sse_list, curve='concave')
    kl.plot_knee()
    plt.grid()
    plt.savefig("kneetest.png")
    plt.show()
    return kl.knee

def main():               
    #train_path = "DAWIDD/flatten_ae_febtrain.csv"
    #train_path = 'DAWIDD/flatten_contrastive_trained_on_feb_feb.csv'
    train_path = 'DAWIDD/flatten_contrastive_trained_on_feb_feb_mpl_train.csv'
    data = pd.read_csv(train_path, index_col=0)
    print(data.head())
    print(len(data))
    skip_columns = ['filename', 'datetime']
    selected_columns = [col for col in data.columns if col not in skip_columns]
    x = data[selected_columns]
    #_ = elbow_test_kmeans(x, max_clusters=50)
    #dimensions = [2, 4, 8, 16, 32, 64, 128, 256]
    #dimensions = [x for x in range (1,257)]
    #sse_lists = super_elbow(data=x, max_clusters=50, pcas=dimensions)
    #sse_lists = torch.load("super_elbow.pickle")
    
    #print(data)
    
    #for sse in sse_lists:
    #    print(sse.values())
    #    for pca, value in sse.items():
    #        plt.plot(range(1, 51), value, label=pca)
            
    #plt.xticks(range(1, 51))
    #plt.xlabel("Number of Clusters (K)")
    #plt.ylabel("SSE")
    #plt.title("SSE versus K")
    #plt.legend()
    #plt.grid()
    #plt.savefig("elbow_test_SSE_pca_ae.png")
    #plt.show()
    
    #pca = PCA(n_components=2, random_state=0)
    #x = pca.fit_transform(x)
    #print(x)
    _ = elbow_test_kmeans(x, 50)
    return
    #val_path = "DAWIDD/flatten_ae_febvalid.csv"
    val_path = "DAWIDD/flatten_contrastive_trained_on_feb_feb_mpl_valid.csv"
    val_data = pd.read_csv(val_path, index_col=0)
    val_x = val_data[selected_columns]
    
    #val_x = pca.transform(val_x)
    
    # tests looks like 10 or 11 clusters might be right
    #sse_list = elbow_test_kmeans(x, 50)
    #sse_list = [23542589.789586496, 21498169.88003995, 20271101.2906938, 19199255.71911375, 18393983.081164885, 17923540.384827383, 17479035.940150872, 17040034.56302915, 16721885.409629248, 16365280.96272025, 16113545.624800304, 15908306.642383002, 15633895.795862336, 15334862.169846395, 15177063.630348988, 14968562.34959302, 14721601.662267905, 14631269.653389055, 14465677.355487859, 14334569.178588955, 14169889.403821517, 14014135.887497248, 13887906.080562595, 13756002.720828438, 13578834.686285611, 13343519.001348566, 13327064.62264193, 13216699.14226864, 13093096.33987166, 13028752.768457849, 12849814.749112425, 12802044.775916185, 12721874.259528043, 12641761.822811725, 12529989.307347499, 12472209.076508986, 12361130.883468427, 12192472.046325097, 12085770.10700225, 12154135.17285408, 12007578.087038137, 11941811.0693445, 11844063.219504602, 11829368.858197747, 11790767.82675452, 11657802.356078733, 11599444.428480672, 11484361.478761718, 11552486.954897039, 11413154.813050613, 11349819.51515079, 11285011.803252805, 11260852.959139965, 11145795.121360749, 11148257.30279571, 11121131.634637676, 11071116.61494945, 10980131.44833719, 10861104.089496389, 10873578.247950397, 10780840.604813367, 10779041.776354631, 10703979.379598876, 10682878.109280555, 10669359.890180964, 10554168.669450104, 10524556.418626586, 10511268.63625113, 10485785.78713731, 10377834.264816605, 10370000.161465142, 10327640.503243506, 10304301.530450685, 10203388.531881172, 10207807.518150952, 10183547.31217829, 10099495.567604478, 9985172.951770643, 10010350.237965653, 9966908.05867517, 9875748.214451373, 9880230.94900097, 9819423.706581477, 9781719.455322297, 9786988.41275289, 9757259.227697162, 9717371.662609592, 9689315.481749278, 9613162.135320593, 9571021.021256408, 9568851.001051936, 9536923.745960288, 9498909.725757392, 9403836.658169324, 9400060.059650356, 9347329.39011464, 9298234.73623897, 9298782.732636448, 9284365.92022549, 9225541.319974992, 9196801.604634438, 9163245.708640866, 9174655.158290228, 9097855.635356009, 9035800.113553206, 9052004.168509273, 8995430.574719038, 8974597.228438623, 8932557.817165624, 8904495.521264518, 8898394.35430907, 8851266.725159692, 8785945.092474071, 8824589.148036422, 8786271.452116773, 8711155.640694259, 8738101.552761767, 8729538.58160787, 8621945.905400395, 8599932.62028914, 8559739.19198725, 8499348.126353431, 8518582.17120952, 8478134.745905094, 8456124.156743532, 8445656.931773083, 8395374.703958169, 8363329.823985386, 8400653.068321673, 8347301.983377381, 8334896.517469459, 8247400.738373799, 8251751.901297129, 8171824.22638035, 8243986.603374852, 8140897.5870841835, 8162412.583988023, 8117434.51704533, 8032871.395770355, 8063653.416676103, 8083400.826968223, 8025258.021095727, 7992329.3197801225, 7958820.90862622, 7924839.555570319, 7926507.119760728, 7914313.811779446, 7868551.1554281935, 7829584.873080404, 7809859.5473348, 7757595.3000766495, 7751678.168503806, 7748404.772891991, 7668215.876621609, 7703399.147485677, 7685946.06008774, 7655603.003087608, 7645277.917434964, 7607409.902415239, 7601865.614733119, 7554070.441685311, 7543826.105586822, 7529776.298814101, 7443571.412158294, 7497671.717774852, 7448378.393409708, 7427897.780492525, 7417701.481170736, 7376430.190534919, 7385733.403040495, 7367873.918994966, 7327609.970568484, 7337830.932276601, 7255298.107703503, 7227413.7847066065, 7257724.23014434, 7268116.635652905, 7215547.183546837, 7217526.999440888, 7182828.018529229, 7134801.322328982, 7128623.4545273855, 7060511.164228563, 7104128.857352745, 7085400.684466925, 7054924.712239813, 7060943.487020973, 7054675.074917628, 6988988.585904969, 7022696.492615294, 6965775.255237714, 6952053.704155999, 6941428.511807198, 6939283.403540819, 6903768.642277358, 6875785.859415138, 6839737.134959315, 6811016.355262996, 6835642.835012058, 6829921.339616537]
    
    #Kk = pick_k(sse_list)
    Kk = 7
    #print(Kk)
    cluster_alg = DataClusterer(K=Kk, data=x)
    
    train_predictions = cluster_alg.predict_cluster(x)
    val_predictions = cluster_alg.predict_cluster(val_x)
    
    data['cluster'] = train_predictions
    data.to_csv("DAWIDD/flatten_mlp_256_febtrain_with_clusters.csv")
    val_data['cluster'] = val_predictions
    val_data.to_csv("DAWIDD/flatten_mlp_256_febval_with_clusters.csv")
    #test = [23542589.7895865, 21498169.880039953, 20271101.2906938, 19199255.71911375, 18393983.08116488, 17923540.384827383, 17479035.940150872, 17040034.563029155, 16721885.409629248, 16365280.962720249, 16113545.624800302, 15908306.642383, 15633895.795862336, 15334862.169846393, 15177063.630348986, 14968562.349593021, 14721601.662267905, 14631269.653389057, 14465677.35548786, 14334569.178588955, 14169889.403821519, 14014135.887497246, 13887906.080562595, 13756002.720828436, 13578834.686285611, 13343519.001348564, 13327064.62264193, 13216699.14226864, 13093096.339871664, 13028752.76845785, 12849814.749112427, 12802044.775916187, 12721874.259528043, 12641761.822811725, 12529989.307347497, 12472209.076508984, 12361130.883468429, 12192472.046325099, 12085770.10700225, 12154135.172854083, 12007578.087038139, 11941811.069344498, 11844063.219504606, 11829368.858197747, 11790767.826754522, 11657802.356078736, 11599444.428480672, 11484361.478761718, 11552486.95489704]
    #k = pick_k(test)
    data_to_write = pd.read_csv("DAWIDD/flatten_mlp_256_febtrain_with_clusters.csv", index_col=0)
    #print(data_to_write.head())
    
    write_cluster_txts(data_to_write)
    return
    img_folder = "Data/images/valid/" #"Data/images/train/"
    file_txt = "Data/Febvalid.txt"
    #Dataloading train 
    filenames = []
    
    with open(file_txt) as reader:
        for filename in reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append(img_folder + filename)
            
    print(len(filenames))
    
    random.seed(0)
    np.random.seed(0)
    
    splits = (np.array_split(filenames, 7))
    
    total = 0
    for split in splits:
        print(len(split))
        total += len(split)
    print(total)
    
    for index, split in tqdm(enumerate(splits), desc="Writing .txts from clusters", total=len(splits)):
        path = f"Data/random_cluster_{index}_valid.txt"
        random.shuffle(split)
        with open(path, 'w') as f:
            for file in split:
                f.write(file + "\n")

if __name__ == '__main__':
    main()
