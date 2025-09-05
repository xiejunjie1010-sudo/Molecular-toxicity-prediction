#!/usr/bin/env python
"""
5-fold HRAN ⊕ 基因/通路可训练嵌入
保存位置：HRAN/save/kg_fp_fold5/

python HRAN/train/train_hran_cv.py \
       --gene_col geneSymbol --pathway_col pathwayId \
       --epochs 1000 --hidden 512
"""

import argparse, json, pickle, numpy as np, pandas as pd, torch, time, glob
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score,
    roc_curve, auc
)
from torch_geometric.nn import RGCNConv
from tqdm import tqdm
import matplotlib.pyplot as plt

# === HRANConv: R-GCN 卷积 + 关系注意力 ==========================
class HRANConv(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_rel, num_bases=8, heads=4):
        super().__init__()
        assert out_dim % heads == 0, "out_dim 必须能被 heads 整除"
        self.rgcn  = RGCNConv(in_dim, out_dim, num_rel, num_bases)
        self.heads = heads
        self.head_dim = out_dim // heads
        self.att = torch.nn.Parameter(torch.Tensor(heads, self.head_dim))
        torch.nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, edge_type):
        h = self.rgcn(x, edge_index, edge_type)                     # [N, out_dim]
        h = h.view(-1, self.heads, self.head_dim)                   # [N, H, d]
        score = (h * self.att).sum(-1)                              # [N, H]
        alpha = torch.softmax(score, dim=1).unsqueeze(-1)           # [N, H, 1]
        # 逐 head 加权后直接展开 → [N, out_dim]
        h_out = (h * alpha).reshape(-1, self.heads * self.head_dim)
        return h_out, alpha.squeeze(-1)                             # α:[N,H]
# ===============================================================

# >>> 受体缩写顺序必须与 label_matrix.npy 列顺序一致
RECEPTORS = [
    "NR-AhR", "NR-AR", "NR-AR-LBD", "NR-Aromatase",
    "NR-ER",  "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5",  "SR-HSE", "SR-MMP", "SR-p53"
]

# ---------------- CLI ----------------
ap = argparse.ArgumentParser()
ap.add_argument('--kg_dir', default='../../data/KG/r-gcn')
ap.add_argument('--epochs', type=int, default=800)
ap.add_argument('--hidden', type=int, default=512)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--embed_dim', type=int, default=64)
ap.add_argument('--gene_col', default='geneSymbol')
ap.add_argument('--pathway_col', default='pathwayId')
args = ap.parse_args()

KG_DIR   = Path(args.kg_dir).resolve()
SAVE_DIR = Path('../save/kg_fp_fold5_HRAN'); SAVE_DIR.mkdir(parents=True, exist_ok=True)
device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- 数据加载 ----------------
edge_index = torch.from_numpy(np.load(KG_DIR/'edge_index.npy')).long()
edge_type  = torch.from_numpy(np.load(KG_DIR/'edge_type.npy')).long()
x          = torch.from_numpy(np.load(KG_DIR/'node_features.npy')).float()
labels     = torch.from_numpy(np.load(KG_DIR/'label_matrix.npy')).float()
masks      = pickle.load(open(KG_DIR/'fold_masks.pkl','rb'))
gene_idx   = torch.from_numpy(np.load(KG_DIR/'gene_idx.npy')).long()
path_idx   = torch.from_numpy(np.load(KG_DIR/'pathway_idx.npy')).long()

num_rel = edge_type.max().item()+1
in_dim  = x.size(1)

# ---------------- HRAN 模型 ----------------
class HRAN(torch.nn.Module):
    def __init__(self, in_dim, h_dim, num_rel, emb_dim, n_gene, n_path):
        super().__init__()
        self.emb_dim = emb_dim
        self.gene_emb = torch.nn.Embedding(n_gene, emb_dim)
        self.path_emb = torch.nn.Embedding(n_path, emb_dim)
        self.conv1 = HRANConv(in_dim+emb_dim, h_dim, num_rel,
                              num_bases=8, heads=4)
        self.conv2 = HRANConv(h_dim, h_dim, num_rel,
                              num_bases=8, heads=4)
        self.pred  = torch.nn.Linear(h_dim, 12)

    def forward(self, x, ei, et, gene_idx, path_idx):
        zeros = torch.zeros(x.size(0), self.emb_dim, device=x.device)
        x = torch.cat([x, zeros], dim=1)
        tail = slice(-self.emb_dim, None)
        x[gene_idx, tail] += self.gene_emb.weight
        x[path_idx, tail] += self.path_emb.weight
        h,_ = self.conv1(x, ei, et)
        h,_ = self.conv2(h, ei, et)
        return self.pred(h)          # logits

# ---------------- 逐受体指标 ----------------
def per_receptor_metrics(y_true, y_prob):
    mask = (y_true != -1)
    col_bac, col_f1, col_auc, col_tpr, col_tnr = [], [], [], [], []

    for k in range(12):
        m = mask[:, k]
        if m.sum()==0:
            col_bac.append(np.nan); col_f1.append(np.nan); col_auc.append(np.nan)
            col_tpr.append(np.nan); col_tnr.append(np.nan)
            continue
        y_t, y_p = y_true[m, k], y_prob[m, k]
        y_pred   = (y_p >= 0.5)

        col_bac.append(balanced_accuracy_score(y_t, y_pred))
        col_f1 .append(f1_score(y_t, y_pred, zero_division=0))
        col_auc.append(roc_auc_score(y_t, y_p))

        tp = ((y_t == 1) & (y_pred == 1)).sum()
        fn = ((y_t == 1) & (y_pred == 0)).sum()
        tn = ((y_t == 0) & (y_pred == 0)).sum()
        fp = ((y_t == 0) & (y_pred == 1)).sum()
        col_tpr.append(tp / (tp + fn + 1e-9))
        col_tnr.append(tn / (tn + fp + 1e-9))

    macro = (np.nanmean(col_bac), np.nanmean(col_f1), np.nanmean(col_auc))
    detail = {
        'Receptor': RECEPTORS,
        'BAC': col_bac, 'F1': col_f1, 'AUC': col_auc,
        'Toxic_Acc': col_tpr, 'NonToxic_Acc': col_tnr
    }
    return macro, detail

# ---------------- 单折训练 ----------------
def train_fold(fold, masks):
    model = HRAN(in_dim, args.hidden, num_rel,
                 args.embed_dim, len(gene_idx), len(path_idx)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    pos=(labels==1).float(); neg=(labels==0).float()
    pos_weight=(neg.sum(0)/(pos.sum(0)+1e-6)).clamp(max=10).to(device)
    criterion=torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    train_m,val_m,test_m=[torch.from_numpy(m).to(device) for m in masks]
    ei,et=edge_index.to(device),edge_type.to(device)
    x_dev=x.to(device); lab=labels.to(device)
    best_auc,best_state=-1,None

    for ep in tqdm(range(1,args.epochs+1), desc=f'Fold{fold}'):
        model.train(); optim.zero_grad()
        logits=model(x_dev,ei,et,gene_idx.to(device),path_idx.to(device))
        loss=(criterion(logits[train_m],lab[train_m])
              [lab[train_m]!=-1]).mean()
        loss.backward(); optim.step()

        if ep%100==0 or ep==args.epochs:
            model.eval()
            with torch.no_grad():
                val_logits = model(x_dev,ei,et,gene_idx.to(device),path_idx.to(device))
                macro,_ = per_receptor_metrics(
                    lab[val_m].cpu().numpy(),
                    torch.sigmoid(val_logits[val_m]).cpu().numpy())
            tqdm.write(f' Fold{fold} Ep{ep} Loss {loss:.3f}  ValAUC {macro[2]:.3f}')
            if macro[2] > best_auc:
                best_auc, best_state = macro[2], model.state_dict()

    # -------- 测试 --------
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(x_dev,ei,et,gene_idx.to(device),path_idx.to(device)))
    macro, detail = per_receptor_metrics(lab[test_m].cpu().numpy(),
                                         prob[test_m].cpu().numpy())

    pd.DataFrame(detail).to_csv(SAVE_DIR/f'detail_fold{fold}.csv', index=False)
    pd.DataFrame([[*macro]], columns=['BAC','F1','AUC']) \
        .to_csv(SAVE_DIR/f'metrics_fold{fold}.csv', index=False)
    torch.save(best_state, SAVE_DIR/f'model_fold{fold}.pt')

    # -------- ROC per receptor + mean --------
    y_true = lab[test_m].cpu().numpy(); y_prob = prob[test_m].cpu().numpy()
    mask = (y_true != -1)
    mean_fpr = np.linspace(0,1,100); tprs=[]
    for k,name in enumerate(RECEPTORS):
        m = mask[:,k]
        if m.sum()==0: continue
        fpr,tpr,_ = roc_curve(y_true[m,k], y_prob[m,k])
        auc_k     = auc(fpr,tpr)
        plt.figure(); plt.plot([0,1],[0,1],'--',c='gray')
        plt.plot(fpr,tpr,label=f'AUC={auc_k:.2f}')
        plt.title(f'ROC Fold{fold} – {name}'); plt.legend()
        plt.savefig(SAVE_DIR/f'ROC_fold{fold}_{name}.png'); plt.close()
        tprs.append(np.interp(mean_fpr,fpr,tpr))
    if tprs:
        mean_tpr=np.mean(tprs,axis=0); mean_tpr[-1]=1.0
        plt.figure(); plt.plot([0,1],[0,1],'--',c='gray')
        plt.plot(mean_fpr,mean_tpr,c='b',lw=2,label=f'AUC={macro[2]:.2f}')
        plt.title(f'ROC Fold{fold} Mean'); plt.legend()
        plt.savefig(SAVE_DIR/f'ROC_fold{fold}_mean.png'); plt.close()

    print(f'Fold{fold} TestAUC {macro[2]:.3f}')
    return pd.DataFrame(detail)

# ---------------- 主流程 ----------------
detail_all=[]
for f in range(5):
    detail_all.append(train_fold(f, masks[f]))

summary_per = pd.concat(detail_all).groupby('Receptor').mean()
summary_per.to_csv(SAVE_DIR/'summary_per_receptor.csv', index=True)

macro_files = sorted(glob.glob(str(SAVE_DIR/'metrics_fold*.csv')))
macro_df = pd.concat([pd.read_csv(f) for f in macro_files])
macro_df.mean().to_frame('Mean').to_csv(SAVE_DIR/'summary_macro.csv')

print('\n=== Per-receptor 5-fold Mean ===')
print(summary_per)
print('\n=== Macro 5-fold Mean ===')
print(macro_df.mean())
