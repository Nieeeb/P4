import json
import pandas as pd
from tqdm import tqdm  

with open("Train.json", 'r') as file:
    data = json.load(file)

images = data.get("images", [])
n_images = len(images)

date_captured_list = []
for img in tqdm(images, desc="Reading dates", unit="img"):
    date_captured_list.append(img.get("date_captured"))

df = pd.DataFrame(data=date_captured_list, columns=["date_captured"])
df.to_csv("date_captured_train.csv", index=False)
