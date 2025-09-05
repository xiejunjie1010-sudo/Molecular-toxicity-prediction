#!/usr/bin/env python
"""
生成 5-fold 训练/验证/测试掩码 (bool[N_nodes])，仅作用于化合物节点。

用法
-----
python split_dataset.py \
       --kg_dir ../../data/KG \
       --fold 5 \
       --compound_file r-gcn/compound_master.csv \
       --seed 42
"""

import argparse, json, random, pickle
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kg_dir', required=True)
    ap.add_argument('--compound_file', default='r-gcn/compound_master.csv')
    ap.add_argument('--fold', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()

def main():
    args = parse()
    kg     = Path(args.kg_dir).resolve()
    outdir = kg/'r-gcn'
    nm     = json.load(open(outdir/'node_id_map.json'))
    N      = len(nm)

    # --- 载入标签矩阵 (已由 build_edges_v2.py 生成) ---
    Y = np.load(outdir/'label_matrix.npy')     # shape (N,12)

    # --- 化合物节点索引列表 ---
    cmp = pd.read_csv(kg/args.compound_file)
    cmp['cid'] = cmp['cid'].astype(str)
    cid2id = {str(k):v for k,v in nm.items()}
    cmp_idx = [cid2id[c] for c in cmp['cid'] if str(c) in cid2id]

    # --- 构造 Stratify 标签: 每个化合物阳性标签个数 ---
    pos_count = (Y[cmp_idx] == 1).sum(1)

    skf = StratifiedKFold(n_splits=args.fold, shuffle=True, random_state=args.seed)
    masks = {}

    for fold, (train_val, test) in enumerate(skf.split(cmp_idx, pos_count)):
        # 再在 train_val 内部 9:1 划分 val
        tv_idx = np.array(cmp_idx)[train_val]
        test_idx = np.array(cmp_idx)[test]

        # Simple random val split preserving size
        random.Random(args.seed+fold).shuffle(tv_idx.tolist())
        val_size = max(1, int(len(tv_idx)*0.1))
        val_idx  = tv_idx[:val_size]
        train_idx= tv_idx[val_size:]

        # 构造 bool mask
        train_mask  = np.zeros(N, dtype=bool); train_mask[train_idx]  = True
        val_mask    = np.zeros(N, dtype=bool); val_mask[val_idx]      = True
        test_mask   = np.zeros(N, dtype=bool); test_mask[test_idx]    = True

        masks[fold] = (train_mask, val_mask, test_mask)

        # 打印统计
        def stat(idx):
            ys = Y[idx]; return (ys==1).sum(), (ys!=-1).sum()
        p_tr, n_tr = stat(train_idx)
        p_te, n_te = stat(test_idx)
        print(f'Fold {fold}: train {len(train_idx)}(+{p_tr}/{n_tr}) '
              f'val {len(val_idx)} test {len(test_idx)}(+{p_te}/{n_te})')

    # 保存
    with open(outdir/'fold_masks.pkl','wb') as f:
        pickle.dump(masks,f)
    print('[split] saved →', outdir/'fold_masks.pkl')

if __name__ == '__main__':
    main()
