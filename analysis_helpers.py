import pandas as pd
import json
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def save_dataframes(name, json_path, output_path):
    images_df = pd.DataFrame()
    annotations_df = pd.DataFrame
    with open(json_path) as json_data:
        data = json.load(json_data)
        images_df = pd.DataFrame(data['images'])
        annotations_df = pd.DataFrame(data['annotations'])
        json_data.close()
    annotations_df.to_csv(output_path + name + '_annotations.csv')
    images_df.to_csv(output_path + name + '_images.csv')



def main():
    val_file = r'/home/xander/Documents/School/p4/P4/Data Analysis/Train.json'
    val_output = 'Data Analysis/DataOutputs/'
    save_dataframes('Train', val_file, val_output)
    
    df = pd.read_csv('Data Analysis/DataOutputs/Valid_images.csv', index_col=0)
    #print(df.columns)
    
    temp = []
    for index, row in df.iterrows():
        temp.append(eval(row['meta']))
    
    df_meta = pd.DataFrame(temp)
    scaler = StandardScaler()
    #scaler.fit_transform(df_meta)
    #print(df_meta.head)
    
    dbsacan = DBSCAN(eps=20, min_samples=20)
    dbsacan.fit(df_meta)
    labels = dbsacan.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    
    df_meta['date_captured'] = df['date_captured']
    
    temp_day = []
    temp_time = []
    for index, row in df_meta.iterrows():
        split = str.split(row['date_captured'], 'T')
        day = np.datetime64(split[0])
        #time = np.datetime64(split[1])
        temp_day.append(day)
        #temp_time.append(time)
    df_meta['day_captured'] = temp_day
    df_meta['time_captured'] = pd.to_datetime(df_meta['date_captured']).dt.time
    
    print(df_meta['time_captured'])

    #plt.scatter(df_meta['date_captured'], df_meta['Min of sunshine latest 10 min'])
    #plt.xlabel("Temperature")
    #plt.ylabel("Humidity")
    #plt.title("Temperature vs. Humidity")
    #plt.savefig("scatter_plot_Temp_v_Humid")
    #plt.show()
    
    sns.scatterplot(data=df_meta, x="Wind Direction", y="Wind Speed", hue="Precipitation latest 10 min")
    #plt.savefig("sns_scatterplot_WindD_v_WindS_Precipation")
    plt.show()

    # Things to plot hahhahahahahah
    # Temp
    # Humidity
    # dato
    # Måned/dag
    
    # Numbers to get
    # Sizes of datasets
    # Number of images
    
    
if __name__ == '__main__':
    main()