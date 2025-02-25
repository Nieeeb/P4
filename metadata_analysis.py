import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Data Analysis/DataOutputs/Valid_annotations.csv")
print(df.head())
unique = df["area"].nunique()
print(f"Number of unique values in 'area': {unique}")


df_filtered = df[df["category_id"] == 1]

area_counts = df_filtered["area"].value_counts().sort_index()

plt.figure(figsize=(12, 6))
sns.barplot(x=area_counts.index, y=area_counts.values)
plt.xticks(rotation=45)
plt.show()


