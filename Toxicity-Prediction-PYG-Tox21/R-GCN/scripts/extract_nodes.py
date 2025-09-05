# R-GCN/scripts/extract_nodes.py
import csv, json, pathlib, re
from pathlib import Path

KG_DIR = Path('../../data/KG')
TRIPLE_CSV = KG_DIR / 'filtered_triples3.csv'   # 改成你的最新文件名

gene_set, path_set = set(), set()

def get_props(prop_str: str):
    try:
        return json.loads(prop_str)
    except Exception:
        return {}

with TRIPLE_CSV.open(encoding='utf-8') as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        # -------- 大小写统一 --------
        row = {k.lower(): v for k, v in row.items()}

        hl   = json.loads(row['headlabels'])
        tl   = json.loads(row['taillabels'])
        hprop = get_props(row['headprops'])
        tprop = get_props(row['tailprops'])

        if 'Gene' in hl:
            gene_set.add(hprop.get('geneSymbol', '').strip())
        if 'Gene' in tl:
            gene_set.add(tprop.get('geneSymbol', '').strip())

        if 'Pathway' in hl:
            path_set.add(hprop.get('pathwayId', '').strip())
        if 'Pathway' in tl:
            path_set.add(tprop.get('pathwayId', '').strip())

# 去掉空字符串
gene_set.discard('')
path_set.discard('')

(KG_DIR / 'gnn_input_genes.csv').write_text(
    'geneSymbol\n' + '\n'.join(sorted(gene_set)), encoding='utf-8'
)
(KG_DIR / 'gnn_input_pathways.csv').write_text(
    'pathwayId\n' + '\n'.join(sorted(path_set)), encoding='utf-8'
)

print('genes', len(gene_set), 'pathways', len(path_set))
