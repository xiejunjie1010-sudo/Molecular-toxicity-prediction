#!/usr/bin/env python
"""
构建 node_type.npy（0=Chemical 1=Gene 2=Pathway 3=Other）
---------------------------------------------------------
▶ 自动解析 filtered_triples.csv 中的 headLabels / tailLabels，
  根据标签为每个节点 ID 赋类型。
▶ 输出 <kg_dir>/node_type.npy  (shape [N] int64)

用法示例
--------
python build_node_type.py \
       --kg_dir ../../data/KG \
       --triples filtered_triples.csv
"""
import argparse, json, ast
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kg_dir', required=True,
                    help='KG 根目录，内含 node_id_map.json')
    ap.add_argument('--triples', default='filtered_triples.csv',
                    help='含 head/headLabels/tail/tailLabels 的三元组文件')
    return ap.parse_args()

# ---------- 主 ----------
def main():
    args = parse_args()
    kg   = Path(args.kg_dir).resolve()
    nid_map = json.load(open(kg/'r-gcn'/'node_id_map.json'))
    N = len(nid_map); node_type = np.full(N, 3, dtype=np.int64)  # 3 = Other

    # label → type_id 规则
    def to_type(labels):
        lbl = [l.lower() for l in labels]
        if 'chemical' in lbl: return 0
        if 'gene'      in lbl: return 1
        if 'pathway'   in lbl: return 2
        return 3

    triples = pd.read_csv(kg/args.triples)
    triples.columns = [c.lower() for c in triples.columns]
    for col in ('head', 'headlabels', 'tail', 'taillabels'):
        if col not in triples.columns:
            raise ValueError(f'missing column {col}')

    # 遍历三元组，标注类型
    for _, row in tqdm(triples.iterrows(),
                       total=len(triples), desc='scan triples'):
        for id_col, lab_col in (('head','headlabels'), ('tail','taillabels')):
            node_id = str(row[id_col])
            labels  = ast.literal_eval(row[lab_col])  # "['Gene','Resource']"
            t = to_type(labels)
            idx = nid_map.get(node_id)
            if idx is not None:
                node_type[idx] = t

    out = kg/'r-gcn'/'node_type.npy'
    np.save(out, node_type)
    uniq, cnt = np.unique(node_type, return_counts=True)
    report = {int(k): int(v) for k,v in zip(uniq,cnt)}
    print(f'[node_type] saved → {out} shape={node_type.shape}')
    print(f'            type counts {{0:Chemical,1:Gene,2:Path,3:Other}} → {report}')

if __name__ == '__main__':
    main()
