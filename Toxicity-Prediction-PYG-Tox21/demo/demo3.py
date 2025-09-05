import random

from rdkit.SimDivFilters import MaxMinPicker
from sklearn.neighbors import NearestNeighbors
import numpy as np
from torch_geometric.data import Dataset, Data
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys

datasets = pd.read_csv('../data/raw/nr-ahr.smiles', sep='\t', header=None, names=["Smiles", "Sample ID", "Label"],
                       encoding='utf-8').reset_index()
num = 0
fps = list()
for smile in datasets['Smiles']:
    mol = Chem.MolFromSmiles(smile)
    if mol:
        num += 1
        fps.append(MACCSkeys.GenMACCSKeys(mol))
datasets = [torch.load(f'../data/processed/nr-ahr_data{i}.pt') for i in range(num)]

positive_data = list()
negative_data = list()
positive_fps = list()
negative_fps = list()

for data, fp in zip(datasets, fps):
    if data.y == 1:
        positive_data.append(data)
        positive_fps.append(fp)
    else:
        negative_data.append(data)
        negative_fps.append(fp)

len_negative = len(negative_data)
len_positive = len(positive_data)

if len_positive >= len_negative:
    def distij(i, j):
        return 1 - DataStructs.TanimotoSimilarity(positive_fps[i], positive_fps[j])


    print("222")
    picker = MaxMinPicker()
    pickIndices = picker.LazyPick(distij, len_positive, len_negative, seed=1)
    picks = [positive_data[x] for x in pickIndices]
    picks.extend(negative_data)
else:
    def distij(i, j):
        return 1 - DataStructs.TanimotoSimilarity(negative_fps[i], negative_fps[j])


    print("111")
    picker = MaxMinPicker()
    pickIndices = picker.LazyPick(distij, len_negative, len_positive, seed=1)
    picks = [negative_data[x] for x in pickIndices]
    picks.extend(positive_data)

print(len(picks))
print(len(datasets))

positive_data = list()
negative_data = list()

for data in picks:
    if data.y == 1:
        positive_data.append(data)
    else:
        negative_data.append(data)
print(len(positive_data), len(negative_data))


