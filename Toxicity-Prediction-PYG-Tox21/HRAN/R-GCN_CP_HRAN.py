#!/usr/bin/env python
"""
对比 R-GCN 与 HRAN 12 受体毒性预测结果
输出：
  comparison_per_receptor.csv  — 五指标逐受体对照 + Δ
  comparison_macro.csv         — BAC / F1 / AUC 宏平均对照 + Δ
  终端打印包含差值与配对 t-test
"""

import pandas as pd
from pathlib import Path
from scipy.stats import ttest_rel

# ----------- 手动指定结果目录 -------------
RGCN_DIR = Path('../R-GCN/save/kg_fp_fold5_12_yn')
HRAN_DIR = Path('../HRAN/save/kg_fp_fold5_HRAN')
# ----------------------------------------

# ========== 1. 逐受体比较（五指标） ==========
cols_detail = ['BAC', 'F1', 'AUC', 'Toxic_Acc', 'NonToxic_Acc']

df_r = pd.read_csv(RGCN_DIR / 'summary_per_receptor.csv').set_index('Receptor')
df_h = pd.read_csv(HRAN_DIR / 'summary_per_receptor.csv').set_index('Receptor')

df_r = df_r[cols_detail].add_prefix('RGCN_')
df_h = df_h[cols_detail].add_prefix('HRAN_')

cmp_detail = df_r.join(df_h)
for m in cols_detail:
    cmp_detail[f'Δ{m}'] = cmp_detail[f'HRAN_{m}'] - cmp_detail[f'RGCN_{m}']

print('\n=== Per-receptor comparison (HRAN − RGCN) ===')
print(cmp_detail.round(4))
cmp_detail.to_csv('comparison_per_receptor.csv')

# ========== 2. 宏平均比较（BAC / F1 / AUC） ==========
cols_macro = ['BAC', 'F1', 'AUC']

macro_r = pd.read_csv(RGCN_DIR / 'summary_macro.csv', index_col=0)['Mean']
macro_h = pd.read_csv(HRAN_DIR / 'summary_macro.csv', index_col=0)['Mean']

# Series → 1×3 DataFrame，再加前缀
macro_r_df = macro_r[cols_macro].to_frame().T.add_prefix('RGCN_')
macro_h_df = macro_h[cols_macro].to_frame().T.add_prefix('HRAN_')

macro_cmp = pd.concat([macro_r_df, macro_h_df], axis=1)
for m in cols_macro:
    macro_cmp[f'Δ{m}'] = macro_cmp[f'HRAN_{m}'] - macro_cmp[f'RGCN_{m}']

print('\n=== Macro (overall) comparison ===')
print(macro_cmp.T.round(4))
macro_cmp.to_csv('comparison_macro.csv')

# ========== 3. Paired t-test（逐受体） ==========
print('\n--- Paired t-test (HRAN vs RGCN, per metric) ---')
for m in cols_detail:
    stat, p = ttest_rel(cmp_detail[f'HRAN_{m}'], cmp_detail[f'RGCN_{m}'])
    print(f'{m:<13}  t = {stat:>6.3f}   p = {p:.4f}')
