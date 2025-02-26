import pandas as pd

df_train = pd.read_csv("Data Analysis/DataOutputs/Train_images.csv")
df_val = pd.read_csv("Data Analysis/DataOutputs/Valid_images.csv")
print(df_train['date_captured'].dtype)

df_train['date_only'] = df_train['date_captured'].str.split('T').str[0] 
df_train['date_only'] = pd.to_datetime(df_train['date_only'])
month_counts_train = df_train['date_only'].dt.to_period('M').value_counts().sort_index()

df_val['date_only'] = df_val['date_captured'].str.split('T').str[0] 
df_val['date_only'] = pd.to_datetime(df_val['date_only'])
month_counts_val = df_val['date_only'].dt.to_period('M').value_counts().sort_index()

print("Train Entries per month:")
print(month_counts_train)

print("Val Entries per month:")
print(month_counts_val)
