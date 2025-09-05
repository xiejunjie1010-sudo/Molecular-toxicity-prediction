import numpy as np, pandas as pd, json, torch
from pathlib import Path

KG   = Path('../../data/KG/r-gcn')      # ← 换成你的目录
node_type = torch.from_numpy(np.load(KG/'node_type.npy')).long()
labels    = np.load(KG/'label_matrix.npy')

has_lbl = (labels != -1).any(1)         # 这些行带标签
tbl = pd.Series(node_type[has_lbl].numpy()).value_counts()
print("\n带标签行在各 node_type 中分布：")
print(tbl.to_string())
