import pandas as pd

df_train = pd.read_csv("Data Analysis/DataOutputs/Train_annotations.csv")
df_val = pd.read_csv("Data Analysis/DataOutputs/Valid_annotations.csv")
#print(df['category_id'].dtype)

#df_klasse = df["category_id"].unique()
df_klasse_train = df_train["category_id"].value_counts().sort_index()
df_klasse_val = df_val["category_id"].value_counts().sort_index()

print(f"Amount of classes within the entire training dataset: {df_klasse_train}")
print(f"Amount of classes within the entire validation dataset: {df_klasse_val}")