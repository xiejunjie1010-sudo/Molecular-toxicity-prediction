# R-GCN/data_utils/merge_compound.py
"""
    合并化合物主表，生成 `KG/compound_master.csv`，列至少包含`cid, smiles, {fingerprint_cols…}, {12 toxicity cols}`
Run:
    python -m data_utils.merge_compound --kg_dir ../../data/KG
"""
import argparse, sys
from pathlib import Path

import pandas as pd


def read_common(common_path: Path) -> pd.DataFrame:
    """读入 common_cids.csv；确保列名统一小写"""
    df = pd.read_csv(common_path)
    df.columns = [c.lower() for c in df.columns]
    if 'cid' not in df.columns or 'smiles' not in df.columns:
        sys.exit('[merge_compound] common_cids.csv 缺少 cid / smiles 列')
    return df


def read_fp(fp_path: Path) -> pd.DataFrame:
    """读入 gnn_input_fp.csv；fingerprint 列保持字符串或数值皆可"""
    df = pd.read_csv(fp_path)
    df.columns = [c.lower() for c in df.columns]
    if 'cid' not in df.columns:
        sys.exit('[merge_compound] gnn_input_fp.csv 缺少 cid 列')
    return df


def merge(common_df: pd.DataFrame, fp_df: pd.DataFrame) -> pd.DataFrame:
    """inner-join，保留两表共同的 cid"""
    merged = common_df.merge(fp_df, on='cid', how='inner', validate='one_to_one')
    # 去重
    merged = merged.drop_duplicates(subset='cid')
    # 重新排序：把 cid/smiles 放前面
    front_cols = ['cid', 'smiles']
    cols = front_cols + [c for c in merged.columns if c not in front_cols]
    return merged[cols]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kg_dir', required=True,
                        help='根目录 KG（包含 common_cids.csv 等）')
    parser.add_argument('--out', default='compound_master.csv',
                        help='输出文件名 (写入 KG/)')
    args = parser.parse_args()

    kg_dir = Path(args.kg_dir).resolve()
    common_csv = kg_dir / 'common_cids.csv'
    fp_csv     = kg_dir / 'gnn_input_fp.csv'
    out_csv    = kg_dir / args.out

    print('[merge_compound] load', common_csv)
    common_df = read_common(common_csv)
    print('[merge_compound] load', fp_csv)
    fp_df = read_fp(fp_csv)

    before = (len(common_df), len(fp_df))
    merged_df = merge(common_df, fp_df)
    after = len(merged_df)
    print(f'[merge_compound] rows: common={before[0]}  fp={before[1]}  merged={after}')

    merged_df.to_csv(out_csv, index=False)
    print('[merge_compound] saved →', out_csv)


if __name__ == '__main__':
    main()
