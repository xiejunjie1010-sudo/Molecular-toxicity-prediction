#!/usr/bin/env python
"""
检查 edge_type.npy:
1. 是否为一维数组
2. 是否只包含 0–21 之间的整数
3. 是否总共 22 个不同关系编号
用法:
    python check_edge_type.py --file ../../data/KG/r-gcn/edge_type.npy
"""
import numpy as np, argparse, pathlib

p = argparse.ArgumentParser()
p.add_argument('--file', default='../../data/KG/r-gcn/edge_type.npy',
               help='edge_type.npy 文件路径')
args = p.parse_args()

path = pathlib.Path(args.file).resolve()
etype = np.load(path)

print(f'加载完成: {path}')
print('形状 shape        :', etype.shape)
print('维度 ndim         :', etype.ndim)
print('数据类型 dtype    :', etype.dtype)
print('最小值 min        :', etype.min())
print('最大值 max        :', etype.max())
print('唯一值个数 n_unique:', np.unique(etype).size)

# --- 判定 ---
assert etype.ndim == 1,        '错误: 不是一维数组!'
assert etype.min() == 0,       '错误: 最小值不是 0!'
assert etype.max() == 21,      '错误: 最大值不是 21!'
assert np.unique(etype).size == 22, '错误: 不包含 22 个不同关系编号!'

print('\n✅ 检查通过——edge_type.npy 符合 0–21 共 22 类关系的要求')
