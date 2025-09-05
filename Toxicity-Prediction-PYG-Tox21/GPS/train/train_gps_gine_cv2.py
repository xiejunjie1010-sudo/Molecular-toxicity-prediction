#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GINE + Block-wise Transformer (仅化合物子图) · 5-fold CV
python train_gps_gine_cv.py --epochs 800 --hidden 128 --heads 4
"""
import argparse, random, sys, numpy as np, torch, pandas as pd
import torch.nn as nn; import torch.nn.functional as F
from pathlib import Path; from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch_geometric.nn import GINEConv
from torch import amp

RECEPTORS = [
    "NR-AhR","NR-AR","NR-AR-LBD","NR-Aromatase",
    "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
    "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"
]

# ───── CLI ─────
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument('--kg_dir',  default='../../data/KG/r-gcn')
ap.add_argument('--epochs',  type=int,   default=1000)
ap.add_argument('--hidden',  type=int,   default=128)
ap.add_argument('--heads',   type=int,   default=4)
ap.add_argument('--layers',  type=int,   default=4)
ap.add_argument('--block',   type=int,   default=512, help='化合物块大小')
ap.add_argument('--dropout', type=float, default=0.1)
ap.add_argument('--lr',      type=float, default=1e-3)
ap.add_argument('--pw_max',  type=float, default=5.0, help='pos_weight 最大值')
ap.add_argument('--seed',    type=int,   default=42)
ap.add_argument('--amp',     action='store_true', help='开启混精度 (显存紧张再用)')
args = ap.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
amp_enable = args.amp

KG   = Path(args.kg_dir).resolve()
SAVE = Path('../save/kg_fp_fold5_GPS-GINE_pwmax{}'.format(args.pw_max))
SAVE.mkdir(parents=True, exist_ok=True)

# ───── 数据加载 ─────
def load(name, dtype=torch.float):
    return torch.from_numpy(np.load(KG/name)).to(DEV).to(dtype)

edge_index = load('edge_index.npy', torch.long)
edge_type  = load('edge_type.npy', torch.long)
x_all      = load('node_features.npy')
labels_all = load('label_matrix.npy')
gene_idx   = load('gene_idx.npy', torch.long)
path_idx   = load('pathway_idx.npy', torch.long)

chem_mask = labels_all[:,0] != -2
chem_idx  = chem_mask.nonzero(as_tuple=True)[0]
y_chem_np = labels_all[chem_idx].cpu().numpy()

# ───── 5-fold 7:2:1 ─────
folds = []
mskf = MultilabelStratifiedKFold(5, shuffle=True, random_state=args.seed)
for trva, te in mskf.split(np.zeros(len(chem_idx)), y_chem_np):
    sp = int(0.77778 * len(trva))
    folds.append((chem_idx[trva[:sp]].clone(),
                  chem_idx[trva[sp:]].clone(),
                  chem_idx[te].clone()))

# ───── 模型层 ─────
class BlockMHA(nn.Module):
    def __init__(self, hidden, heads, drop):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=drop, batch_first=True)
    def forward(self, x):
        if x.size(0) <= 1: return x
        h,_ = self.attn(x[None], x[None], x[None], need_weights=False)
        return x + h.squeeze(0)

class LGBlock(nn.Module):
    def __init__(self, hidden, heads, drop):
        super().__init__()
        mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.local = GINEConv(mlp, edge_dim=hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.mha   = BlockMHA(hidden, heads, drop)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn   = nn.Sequential(nn.Linear(hidden, hidden*2),
                                   nn.ReLU(), nn.Dropout(drop),
                                   nn.Linear(hidden*2, hidden))
    def forward(self, x, ei, ea, chem_idx, B):
        x = self.norm1(x + self.local(x, ei, ea))
        for i in range(0, chem_idx.size(0), B):
            sl = chem_idx[i:i+B]
            x[sl] = self.mha(x[sl])
        x[chem_idx] = x[chem_idx] + self.ffn(self.norm2(x[chem_idx]))
        return x

class Model(nn.Module):
    def __init__(self, in_dim, hidden, heads, layers, num_rel, chem_idx, B):
        super().__init__()
        self.chem_idx, self.B = chem_idx, B
        self.node_proj = nn.Linear(in_dim, hidden) if in_dim!=hidden else nn.Identity()
        self.rel_emb   = nn.Embedding(num_rel, hidden)
        self.gene_emb  = nn.Parameter(torch.zeros(1, hidden))
        self.path_emb  = nn.Parameter(torch.zeros(1, hidden))
        self.blocks    = nn.ModuleList(LGBlock(hidden, heads, args.dropout)
                                       for _ in range(layers))
        self.out = nn.Linear(hidden, 12)
    def forward(self, x, ei, et):
        x = self.node_proj(x).clone()
        x[gene_idx] += self.gene_emb
        x[path_idx] += self.path_emb
        ea = self.rel_emb(et)
        for blk in self.blocks:
            x = blk(x, ei, ea, self.chem_idx, self.B)
        return self.out(x)

num_rel = int(edge_type.max()) + 1
chem_idx_tensor = chem_idx.to(DEV)

# ───── 指标 ─────
def metrics(y_t, y_p):
    mask = (y_t!=-1)
    cols = {'Receptor':RECEPTORS,'BAC':[],'F1':[],'AUC':[],
            'Toxic_Acc':[],'NonToxic_Acc':[]}
    for k in range(12):
        m = mask[:,k]
        if m.sum()==0:
            for key in list(cols.keys())[1:]: cols[key].append(np.nan); continue
        yt, yp = y_t[m,k], y_p[m,k]; yhat = yp>=.5
        cols['BAC'].append(balanced_accuracy_score(yt,yhat))
        cols['F1'] .append(f1_score(yt,yhat,zero_division=0))
        cols['AUC'].append(np.nan if np.unique(yt).size<2 else roc_auc_score(yt,yp))
        tp=((yt==1)&(yhat==1)).sum(); fn=((yt==1)&(yhat==0)).sum()
        tn=((yt==0)&(yhat==0)).sum(); fp=((yt==0)&(yhat==1)).sum()
        cols['Toxic_Acc'].append(tp/(tp+fn+1e-9))
        cols['NonToxic_Acc'].append(tn/(tn+fp+1e-9))
    macro = (np.nanmean(cols['BAC']), np.nanmean(cols['F1']), np.nanmean(cols['AUC']))
    return macro, pd.DataFrame(cols)

# ───── 训练单折 ─────
def train_fold(fid, masks):
    net = Model(x_all.size(1), args.hidden, args.heads, args.layers,
                num_rel, chem_idx_tensor, args.block).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    scaler = amp.GradScaler(enabled=amp_enable)

    tr, val, te = [m.to(DEV) for m in masks]
    pos = (labels_all==1).float(); neg = (labels_all==0).float()
    pos_w = (neg.sum(0)/(pos.sum(0)+1e-6)).clamp(max=args.pw_max).to(DEV)
    crit = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_w)

    best_auc, best_state = -1, None
    ei, et = edge_index, edge_type

    for ep in tqdm(range(1,args.epochs+1), desc=f'Fold{fid}'):
        net.train(); opt.zero_grad(set_to_none=True)
        with amp.autocast(enabled=amp_enable, device_type='cuda'):
            logit = net(x_all, ei, et)
            loss  = crit(logit[tr], labels_all[tr])
            loss  = loss[labels_all[tr]!=-1].mean()

        if torch.isnan(loss):
            torch.save(net.state_dict(), SAVE/f'model_nan_fold{fid}.pt')
            sys.exit(f'NaN detected at epoch {ep}. Saved weights and terminated.')

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt); scaler.update()

        if ep%100==0 or ep==args.epochs:
            net.eval()
            with amp.autocast(enabled=amp_enable, device_type='cuda'):
                prob = torch.sigmoid(net(x_all, ei, et))
            val_auc = metrics(labels_all[val].detach().cpu().numpy(),
                              prob[val].detach().cpu().numpy())[0][2]
            tqdm.write(f'  Ep{ep} Loss {loss:.3f} ValAUC {val_auc:.3f}')
            if val_auc>best_auc:
                best_auc, best_state = val_auc, net.state_dict()

    # ─── 测试 ───
    net.load_state_dict(best_state); net.eval()
    torch.save(best_state, SAVE/f'model_fold{fid}.pt')        # ★ 保存最佳权重
    with amp.autocast(enabled=amp_enable, device_type='cuda'):
        prob = torch.sigmoid(net(x_all, ei, et))
    macro, detail = metrics(labels_all[te].detach().cpu().numpy(),
                            prob[te].detach().cpu().numpy())
    detail.to_csv(SAVE/f'detail_fold{fid}.csv', index=False)
    pd.DataFrame([macro], columns=['BAC','F1','AUC']) \
      .to_csv(SAVE/f'metrics_fold{fid}.csv', index=False)
    print(f'Fold{fid} TestAUC {macro[2]:.3f}')
    return detail

# ───── 主流程 ─────
detail_all=[]
for f,(tr,val,te) in enumerate(folds):
    detail_all.append(train_fold(f,(tr,val,te)))

pd.concat(detail_all).groupby('Receptor').mean() \
  .to_csv(SAVE/'summary_per_receptor.csv')
macro_df = pd.concat([pd.read_csv(f)
                      for f in sorted(SAVE.glob('metrics_fold*.csv'))])
macro_df.mean().to_frame('Mean') \
        .to_csv(SAVE/'summary_macro.csv')

print('\n=== Per-receptor 5-fold Mean ===')
print(pd.read_csv(SAVE/'summary_per_receptor.csv'))
print('\n=== Macro 5-fold Mean ===')
print(macro_df.mean())
