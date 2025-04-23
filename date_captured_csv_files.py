import json
import pandas as pd

with open("/ceph/project/DAKI4-thermal-2025/harborfrontv2/Train.json", 'r') as file:
    data = json.load(file)

n_images = len(data.get("images"))


date_captured_list = [data.get("images")[i].get("date_captured") for i in range(n_images)]

df = pd.DataFrame(data=date_captured_list, columns=["date_captured"])
df.to_csv("date_captured_train.csv")
