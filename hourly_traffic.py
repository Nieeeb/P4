import pandas as pd
import matplotlib.pyplot as plt

df_val = pd.read_csv("Data Analysis/DataOutputs/Valid_images.csv")
df_train = pd.read_csv("Data Analysis/DataOutputs/Train_images.csv")


df_val['datetime'] = pd.to_datetime(df_val['date_captured'])
df_val['hour'] = df_val['datetime'].dt.hour
hour_counts_val = df_val['hour'].value_counts().sort_index()
print(hour_counts_val)

df_train['datetime'] = pd.to_datetime(df_train['date_captured'])
df_train['hour'] = df_train['datetime'].dt.hour
hour_counts_train = df_train['hour'].value_counts().sort_index()
print(hour_counts_train)

plt.figure(figsize=(10, 5))
plt.bar(hour_counts_train.index, hour_counts_train.values, color='skyblue')
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Entries")
plt.title("Number of Entries per Hour TRAIN")
plt.xticks(range(24))
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(hour_counts_val.index, hour_counts_val.values, color='red')
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Entries")
plt.title("Number of Entries per Hour VAL")
plt.xticks(range(24))
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
