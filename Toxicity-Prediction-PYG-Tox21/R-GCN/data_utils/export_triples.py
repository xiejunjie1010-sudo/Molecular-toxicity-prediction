#!/usr/bin/env python
"""
Connects to Neo4j, streams all (h)-[r]->(t) triples, and writes
CSV with columns:
  head,relation,tail,headLabels,tailLabels,headProps,tailProps

Usage
-----
python export_triples.py \
       --uri bolt://localhost:7687 \
       --user neo4j --password ****** \
       --out data/KG/filtered_triples3.csv \
       --batch_size 50000
"""

import argparse, csv, json
from pathlib import Path

from neo4j import GraphDatabase
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--uri', default='bolt://localhost:7687')
    ap.add_argument('--user', default='neo4j')
    ap.add_argument('--password', required=True)
    ap.add_argument('--out', required=True, help='output CSV path')
    ap.add_argument('--batch_size', type=int, default=50_000)
    ap.add_argument('--db', default=None, help='Neo4j database name (4.x+)')
    return ap.parse_args()


COLUMNS = ['head', 'relation', 'tail',
           'headLabels', 'tailLabels', 'headProps', 'tailProps']


def main():
    args = parse_args()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))

    # -------- 获取总关系数 --------
    with driver.session(database=args.db) as sess:
        total = sess.run("MATCH ()-[r]->() RETURN count(r)").single()[0]
    print(f'[export] total relationships: {total:,}')

    query = (
        "MATCH (h)-[r]->(t) "
        "RETURN h AS h, type(r) AS rel, t AS t "
        "SKIP $skip LIMIT $limit"
    )

    with driver.session(database=args.db) as sess, \
         out_path.open('w', newline='', encoding='utf-8') as f:

        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()

        pbar = tqdm(total=total, unit='edge')
        for skip in range(0, total, args.batch_size):
            records = sess.run(query, skip=skip, limit=args.batch_size)
            for rec in records:
                h, t = rec['h'], rec['t']
                # ---------- 关键修正：labels 转 list ----------
                row = dict(
                    head        = h.id,
                    relation    = rec['rel'],
                    tail        = t.id,
                    headLabels  = json.dumps(list(h.labels)),   # frozenset → list
                    tailLabels  = json.dumps(list(t.labels)),   # frozenset → list
                    headProps   = json.dumps(dict(h), default=str),
                    tailProps   = json.dumps(dict(t), default=str),
                )
                writer.writerow(row)
                pbar.update(1)
        pbar.close()

    driver.close()
    print('[export] saved →', out_path)


if __name__ == '__main__':
    main()
