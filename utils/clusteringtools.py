import pandas as pd
#from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
import cudf
from cuml.cluster import KMeans


def elbow_test_kmeans(data, max_clusters):
    random_state = 0
    rss_list = []
    
    for k in tqdm(range(1, max_clusters), desc="Fitting Cluster Sizes", total=max_clusters):
        kmeans = KMeans(n_clusters=k,
                        random_state=random_state,
                        n_init=10,
                        init='scalable-k-means++'
                        )
        kmeans.fit(data)
        rss = kmeans.score(data)
        rss_list.append(rss)
    
    print(rss_list)
    plt.plot(range(1, max_clusters), rss_list)
    plt.xticks(range(1, max_clusters))
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("RSS")
    plt.title("RSS versus K")
    plt.grid()
    plt.savefig("elbow_test_rss.png")
    plt.show()

def main():
    path = "Data/DAWIDD/flatten_ae.csv"
    data = pd.read_csv(path, index_col=0)
    #data = cudf.read_csv(path, index_col=0)
    data = cudf.from_pandas(data)
    skip_columns = ['filename', 'datetime']
    selected_columns = [col for col in data.columns if col not in skip_columns]
    x = data[selected_columns]
    
    elbow_test_kmeans(x, 30)

if __name__ == '__main__':
    main()
