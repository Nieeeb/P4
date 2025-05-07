import json

with open(r'Test.json', 'r') as f:
    test = json.load(f) 

print(f'Test: {len(list(test.get('annotations')))}')


with open(r'Train.json', 'r') as f:
    train = json.load(f) 

print(f'Train: {len(list(train.get('annotations')))}')

with open(r'Valid.json', 'r') as f:
    valid = json.load(f) 

print(f'Valid: {len(list(valid.get('annotations')))}')