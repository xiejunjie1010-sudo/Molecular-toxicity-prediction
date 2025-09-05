#!/usr/bin/env python
"""
从 filtered_triples3.csv 生成：
  edge_index.npy, edge_type.npy, node_id_map.json, rel_id_map.json, label_matrix.npy
逻辑：
  • Chemical 节点键 = PubChem CID (xrefPubchemCID)
  • Gene 节点键     = geneSymbol
  • Pathway 节点键  = pathwayId
  • 其它节点键     = internal neo4j id

  python build_edges.py \
       --kg_dir ../../data/KG \
       --triples filtered_triples3.csv \
       --compound_file r-gcn/compound_master.csv \
       --add_inverse
"""

import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm

KEY_HEAD_PROPS = 'headProps'
KEY_TAIL_PROPS = 'tailProps'

CID_PAT = re.compile(r'"xrefPubchemCID"\s*:\s*"(\d+)"', re.I)
GENE_PAT = re.compile(r'"geneSymbol"\s*:\s*"([^"]+)"', re.I)
PATH_PAT = re.compile(r'"pathwayId"\s*:\s*"([^"]+)"', re.I)

def node_key(internal_id: str, prop_str: str) -> str:
    """根据属性串返回统一节点键"""
    if m := CID_PAT.search(prop_str):
        return m.group(1)           # Chemical → PubChem CID
    if m := GENE_PAT.search(prop_str):
        return m.group(1).upper()   # Gene → geneSymbol
    if m := PATH_PAT.search(prop_str):
        return m.group(1).upper()   # Pathway → pathwayId
    return internal_id              # 其它 → Neo4j internal id

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kg_dir', required=True)
    ap.add_argument('--triples', required=True)
    ap.add_argument('--compound_file', default='r-gcn/compound_master.csv')
    ap.add_argument('--add_inverse', action='store_true')
    return ap.parse_args()

def main():
    args = parse_args()
    kg_dir = Path(args.kg_dir).resolve()
    out_dir = kg_dir / 'r-gcn'
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(kg_dir/args.triples)
    df.columns = [c.lower() for c in df.columns]
    required = {'head','relation','tail', KEY_HEAD_PROPS.lower(), KEY_TAIL_PROPS.lower()}
    if not required.issubset(df.columns): sys.exit('columns not match')

    node2id, rel2id = {}, {}; src, dst, etype = [], [], []
    def gid(x):
        if x not in node2id: node2id[x] = len(node2id)
        return node2id[x]
    def rid(r):
        if r not in rel2id: rel2id[r] = len(rel2id)
        return rel2id[r]

    print('[edges] mapping nodes / relations')
    for h, r, t, hp, tp in tqdm(df[['head','relation','tail', KEY_HEAD_PROPS.lower(), KEY_TAIL_PROPS.lower()]].itertuples(index=False),
                                total=len(df)):
        h_key = node_key(str(h), hp)
        t_key = node_key(str(t), tp)
        src.append(gid(h_key)); dst.append(gid(t_key)); etype.append(rid(r))
        if args.add_inverse:
            src.append(gid(t_key)); dst.append(gid(h_key)); etype.append(rid(f'{r}_inv'))

    np.save(out_dir/'edge_index.npy', np.vstack([src,dst]).astype(np.int64))
    np.save(out_dir/'edge_type.npy',  np.array(etype, dtype=np.int64))
    json.dump(rel2id,  open(out_dir/'rel_id_map.json','w'),  indent=2)
    json.dump(node2id, open(out_dir/'node_id_map.json','w'), indent=2)
    print('[edges] edge_index', len(src), '| relations', len(rel2id), '| nodes', len(node2id))

    # -------- 生成 label_matrix.npy --------
    cmp = pd.read_csv(kg_dir/args.compound_file)
    cmp['cid'] = cmp['cid'].astype(str)
    tox_cols = [c for c in cmp.columns if c.lower().startswith(('nr','sr'))]
    Y = np.full((len(node2id), 12), -1., np.float32)
    miss=0
    for cid,*vals in cmp[['cid']+tox_cols].itertuples(index=False):
        nid = node2id.get(cid)
        if nid is None: miss+=1; continue
        Y[nid] = np.array(vals, np.float32)
    np.save(out_dir/'label_matrix.npy', Y)
    print('[edges] label_matrix', Y.shape, 'miss', miss)

if __name__ == '__main__':
    main()
