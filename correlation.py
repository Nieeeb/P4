import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from tqdm import tqdm
import torch
from DAWIDD.inference_csv import extract_datetimes
from collections import defaultdict
import random
import calendar
import numpy as np

def extract_granger_results(results):
    p_value = results['1']['']


def write_month_txts(data: pd.DataFrame, total_files):
    random.seed(0)
    
    monthly_data = defaultdict(list)
    for index, row in tqdm(data.iterrows(), desc="Grouping Data by Month", total=(len(data))):
        month = row['dates'].month
        monthly_data[month].append(row['files'])
    totals = 0
    perc = 0
    for month, file_names in tqdm(monthly_data.items(), desc="Writing .txts for months", total=len(monthly_data)):
        totals += len(file_names)
        perc += len(file_names) / total_files * 100
        print(f"Num files for {calendar.month_name[month]}: {len(file_names)}, which is {len(file_names) / total_files * 100}%")
        path = f"Data/test_{calendar.month_name[month]}.txt"
        random.shuffle(file_names)
        with open(path, 'w') as f:
            for file in file_names:
                f.write(file + "\n")
    print(f"Sanity Check: {totals}, {perc}")

def main():
    
    #cluster_states = torch.load('/home/nieb/Downloads/clusters/clusters', weights_only=False)
    #cluster6 = torch.load('/home/nieb/Downloads/clusters/latest', weights_only=False)
    
    #print(cluster6['model'])
    #print(cluster_states['clusters'][6]['model'])
    #print(cluster_states.keys())
    
    #cluster_states['clusters'][6]['model'] = cluster6['model']
    #torch.save(cluster_states, '/home/nieb/Downloads/clusters/combined')
    
    #return
    results = []
    baseline = {"test": "Baseline", "results": pd.read_csv("final_models/baseline_feb_results.csv", index_col=0)}
    results.append(baseline)
    fulltrain = {"test": "Trained On Full Train Set", "results": pd.read_csv("final_models/fulltrain_results.csv", index_col=0)}
    results.append(fulltrain)
    random = {"test": "Random Clusters", "results": pd.read_csv("final_models/random_results.csv", index_col=0)}
    results.append(random)
    ae_pca = {"test": "AE Clusters (2 PCA)", "results": pd.read_csv("final_models/ae_pca_results.csv", index_col=0)}
    results.append(ae_pca)
    mlp_pca = {"test": "Contrastive Clusters (2 PCA)", "results": pd.read_csv("final_models/mlp_pca_results.csv", index_col=0)}
    results.append(mlp_pca)
    ae = {"test": "AE Clusters (256 dim)", "results": pd.read_csv("final_models/ae_256_results.csv", index_col=0)}
    results.append(ae)
    mlp = {"test": "Contrastive Clusters (256 dim)", "results": pd.read_csv("final_models/mlp_256_results.csv", index_col=0)}
    results.append(mlp)
    
    #    print(sse.values())
    #    for pca, value in sse.items():
    #        plt.plot(range(1, 51), value, label=pca)
    
    key = 'mean_ap'
    
    # 1. Load your Excel file (replace with your actual path)
    #df = pd.read_excel(r'C:\Users\Victor Steinrud\Documents\DAKI\4. Semester\P4\Book1.xlsx')
    plt.figure(figsize=(10, 6))
    for result in results:
        df = result['results']
    #    # 2. Assume rows correspond to January→August in order
    #    #    Create a month index 1–8 so February = 2
        df['month_idx'] = np.arange(1, len(df) + 1)

        # 3. Prepare for S_LTS calculation: exclude February
        df_eval = df[df['month_idx'] != 2].copy()
        df_eval['d_m'] = np.abs(df_eval['month_idx'] - 2)

        # 4. Total weight W = sum of distances
        W = df_eval['d_m'].sum()  # should be 22 for Jan–Aug

        # 5. Specify the columns to process
        columns = [
            key
            ]

        # 6. Compute S_LTS for each column
        stability_scores = {}
        for col in columns:
            # weighted average of the mAPs
            stability_scores[col] = (df_eval['d_m'] * df_eval[col]).sum() / W
            print(f"SC: {result['test']}: {stability_scores[col]}")

        # 7. Define x-axis labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']

        # 8. Plot all series with their S_LTS in the legend
        #plt.figure(figsize=(10, 6))
        for col in columns:
            plt.plot(
                month_labels,
                df[col],
                marker='o',
                label=f"{result['test']} (ANPS={stability_scores[col]:.4f})"
            )

        plt.xlabel('Month')
        plt.ylabel('COCO mAP')
        plt.title('Monthly COCO mAP for the detectors, with distance‐weighted stability scores')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
    
    plt.savefig(f"test_coco_map_stability.png")
    plt.show()

    
    
    
    
    
    
    return
    
    plt.figure(figsize=(10, 6))
    for result in results:
        plt.plot(result['results'][key], label=result['test'])
            
    plt.xticks(np.linspace(0,7,8), ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August'])
    plt.xlabel("Month")
    plt.ylabel(f"COCO mAP")
    plt.title(f"COCO mAP Scores Across Months")
    plt.legend()
    plt.grid()
    plt.savefig(f"test_cocomap.png")
    plt.show()

    return
    file_txt = "Data/test.txt"
    img_folder = "Data/images/test/"
    filenames = []
    
    with open(file_txt) as reader:
        for filename in reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append(img_folder + filename)
    files = pd.Series(filenames)
    datetimes = pd.Series(extract_datetimes(filenames))
    
    df = pd.DataFrame()
    df['files'] = files
    df['dates'] = datetimes
    total_files = len(filenames) - 120
    write_month_txts(df, total_files)
    return
    path = 'Book1(1).xlsx'
    
    data = pd.read_excel(path)
    map_score_feb = data.pop('COCO mAP')
    coco_map_full = data.pop('COCO mAP FullTrainSet')
    data = data.drop('Month', axis=1)
    data = data[['Temperature', 'Humidity']]
    print(data.head())
    
    maxlag = 2
    results = {}
    for column_name, column_data in tqdm(data.items(), desc="Calculating Granger causaility for each column", total=len(data.columns)):
        df = pd.DataFrame(data=column_data)
        df['target'] = map_score_feb
        print(f"Results for {column_name} on feb:")
        gc_feb = grangercausalitytests(df, maxlag)
        
        causality = {
            'feb': gc_feb
        }
        results[column_name] = causality
    
    print(results)
    



if __name__ == "__main__":
    main()