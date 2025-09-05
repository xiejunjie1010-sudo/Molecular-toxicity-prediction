#!/usr/bin/env python
"""
生成 node_features.npy (+ gene_idx.npy / pathway_idx.npy)
   • Chemical → 指纹矩阵 (可 SVD)
   • Gene / Pathway → 全 0 (训练时可训练嵌入)

   python build_node_features.py \
       --kg_dir ../../data/KG \
       --compound_file r-gcn/compound_master.csv \
       --gene_file gnn_input_genes.csv \
       --pathway_file gnn_input_pathways.csv \
       --fp_pca 256
"""

import argparse, json, ast, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--kg_dir', required=True)
    p.add_argument('--compound_file', default='r-gcn/compound_master.csv')
    p.add_argument('--gene_file',     default='gnn_input_genes.csv')
    p.add_argument('--pathway_file',  default='gnn_input_pathways.csv')
    p.add_argument('--fp_pca', type=int, default=256)
    p.add_argument('--gene_col',     default='geneSymbol')
    p.add_argument('--pathway_col',  default='pathwayId')
    return p.parse_args()


def main():
    a = parse()
    kg = Path(a.kg_dir).resolve()
    out = kg / 'r-gcn'
    out.mkdir(exist_ok=True)

    node_map = json.load(open(out / 'node_id_map.json'))
    N = len(node_map)
    print('[feat] nodes', N)

    # -------- 化合物指纹 --------
    cmp = pd.read_csv(kg / a.compound_file)
    cmp.columns = [c.lower() for c in cmp.columns]
    fp_cols = [c for c in cmp.columns if c.startswith('fp')]
    if not fp_cols:
        sys.exit('fingerprint columns not found')

    fps = []
    for col in tqdm(fp_cols, desc='parse_fp_cols'):
        col_series = cmp[col]
        if np.issubdtype(col_series.dtype, np.number):
            arr = col_series.values.astype(np.float32).reshape(-1, 1)
        else:
            arr = col_series.apply(lambda s: np.array(ast.literal_eval(s), np.float32)).to_list()
            arr = np.vstack(arr)
        fps.append(arr)

    fp = np.hstack(fps)
    print('[feat] raw', fp.shape)
    if a.fp_pca > 0 and fp.shape[1] > a.fp_pca:
        fp = TruncatedSVD(a.fp_pca, random_state=1).fit_transform(fp)

    F = fp.shape[1]
    X = np.zeros((N, F), np.float32)

    cid2id = {str(k): v for k, v in node_map.items()}
    miss = 0
    for cid, vec in zip(cmp['cid'].astype(str), fp):
        nid = cid2id.get(cid)
        if nid is None:
            miss += 1
            continue
        X[nid] = vec
    print('[feat] compound mapped', len(cmp) - miss, 'miss', miss)

    # -------- Gene / Pathway 索引 --------
    def idx_list(csv_path, col, label):
        # 只读取第一列，跳过格式异常行
        df = pd.read_csv(csv_path,
                         usecols=[0],
                         header=0,
                         names=[col],
                         on_bad_lines='skip',
                         engine='python')
        ids = []
        for key in tqdm(df[col].astype(str), desc=f'map_{label}', leave=False):
            nid = cid2id.get(key.upper())
            if nid is not None:
                ids.append(nid)
        return np.array(ids, np.int64)

    gene_idx    = idx_list(kg / a.gene_file,     a.gene_col,    'gene')
    pathway_idx = idx_list(kg / a.pathway_file,  a.pathway_col, 'pathway')

    # -------- 保存 --------
    np.save(out / 'node_features.npy', X)
    np.save(out / 'gene_idx.npy', gene_idx)
    np.save(out / 'pathway_idx.npy', pathway_idx)
    print('[feat] saved node_features', X.shape,
          '| gene', len(gene_idx), 'pathway', len(pathway_idx))


if __name__ == '__main__':
    main()