import numpy as np
import pandas as pd
from pathlib import Path

KG = Path('../../data/KG/r-gcn')        # ← 按需改成你的真实路径
lbl = np.load(KG/'label_matrix.npy')      # shape=(N_chem, 12)

print('矩阵形状          :', lbl.shape)
print('取值种类          :', np.unique(lbl))
# -1 = 缺失 / 未测, 0 = 阴性, 1 = 阳性

# ① 每行有效标签个数
row_valid = (lbl != -1).sum(1)              # shape (N_chem,)
print('\n=== 每条化合物的有效标签数 ===')
print('最小 / 中位 / 最大:', row_valid.min(),
      np.median(row_valid), row_valid.max())
print('完全没标签的化合物数:', (row_valid == 0).sum())

# 如需查看哪些行全空标签:
row0 = np.where(row_valid == 0)[0]
print('示例空标签行 idx 前 10 个:', row0[:10])

# ② 每列取值计数
col_cnt = {
    'neg(0)': (lbl == 0).sum(0),
    'pos(1)': (lbl == 1).sum(0),
    'missing(-1)': (lbl == -1).sum(0)
}
cnt_df = pd.DataFrame(col_cnt)
cnt_df.index = [f'label_{i}' for i in range(lbl.shape[1])]
print('\n=== 每个标签列计数 ===')
print(cnt_df)

# ③ 如想把统计保存成 CSV 方便查看：
cnt_df.to_csv('label_column_counts.csv')
pd.DataFrame({'chem_idx': np.arange(lbl.shape[0]),
              'valid_cnt': row_valid}).to_csv(
              'label_row_valid_counts.csv', index=False)
print('\n已导出 2 个 CSV：column / row 统计')
