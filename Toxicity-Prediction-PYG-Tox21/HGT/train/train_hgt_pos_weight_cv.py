#!/usr/bin/env python
"""
增加了奖励机制来解决数据不均衡问题
HGT on Tox21 —— 5-fold CV   (with class-imbalance reward: label-wise pos_weight)

输出格式同 R-GCN: detail_fold*, metrics_fold*, summary_per_receptor.csv,
summary_macro.csv, 以及各折最佳模型与预测。

python train_hgt_pos_weight_cv.py --epochs 1000 --hidden 128 --lr 5e-4 --embed_dim 32
"""

import argparse, random, numpy as np, torch, gc
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from torch_geometric.nn import HGTConv
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# 与 label_matrix 列顺序一致
RECEPTORS = [
    "NR-AhR","NR-AR","NR-AR-LBD","NR-Aromatase",
    "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
    "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"
]

# ───────── CLI ─────────
ap = argparse.ArgumentParser()
ap.add_argument('--kg_dir',    default='../../data/KG/r-gcn')
ap.add_argument('--save_dir',  default='../../HGT/save/kg_fp5_HGT_pos_weight')
ap.add_argument('--epochs',    type=int,   default=800)
ap.add_argument('--hidden',    type=int,   default=128)
ap.add_argument('--embed_dim', type=int,   default=32)
ap.add_argument('--heads',     type=int,   default=2)
ap.add_argument('--lr',        type=float, default=5e-4)
ap.add_argument('--seed',      type=int,   default=42)
ap.add_argument('--min_pos',   type=int,   default=1)
ap.add_argument('--chem_type', type=int,   default=-1)
args = ap.parse_args()

SAVE_DIR = Path(args.save_dir); SAVE_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ───────── 读取 KG ─────────
KG = Path(args.kg_dir).resolve()
edge_index = torch.from_numpy(np.load(KG/'edge_index.npy')).long()
edge_type  = torch.from_numpy(np.load(KG/'edge_type.npy')).long()
x_all      = torch.from_numpy(np.load(KG/'node_features.npy')).float()
node_type  = torch.from_numpy(np.load(KG/'node_type.npy')).long()
labels_np  = np.load(KG/'label_matrix.npy')

# ───────── chem_type ─────────
CHEM_T = args.chem_type if args.chem_type >= 0 \
         else np.bincount(node_type[(labels_np!=-1).any(1)].numpy()).argmax()
print(f"[INFO] chem_type = {CHEM_T}")

# ───────── metadata ─────────
nt_names = ['chem','gene','path']+[f'oth{k}' for k in range(3,int(node_type.max())+1)]
chem_key = nt_names[CHEM_T]

x_dict_cpu = {n: x_all[(node_type==i)] for i,n in enumerate(nt_names)}
g2l = {i:{g.item():j for j,g in enumerate((node_type==i).nonzero(as_tuple=True)[0])}
       for i in range(len(nt_names))}

edge_dict_gpu={}
for h,r,t in zip(edge_index[0],edge_type,edge_index[1]):
    s,d=node_type[h].item(),node_type[t].item()
    key=(nt_names[s],f'rel{r.item()}',nt_names[d])
    edge_dict_gpu.setdefault(key,[[],[]])
    edge_dict_gpu[key][0].append(g2l[s][h.item()])
    edge_dict_gpu[key][1].append(g2l[d][t.item()])
edge_dict_gpu={k:torch.tensor(v,device=DEV) for k,v in edge_dict_gpu.items()}
metadata=(list(x_dict_cpu.keys()),list(edge_dict_gpu.keys()))

# ───────── usable chems ─────────
chem_glb=(node_type==CHEM_T).nonzero(as_tuple=True)[0]
has_lbl =(labels_np!=-1).any(1)
chem_idx=chem_glb[has_lbl[chem_glb]]
print(f"[INFO] usable chems: {len(chem_idx)}/{len(chem_glb)}")

chem_loc=torch.tensor([g2l[CHEM_T][int(g)] for g in chem_idx],device=DEV)
y_full  =torch.from_numpy(labels_np[chem_idx]).float().to(DEV)
y_bin   =(labels_np[chem_idx]==1).astype(int)

# ───────── stratified folds ─────────
folds=[]
outer=MultilabelStratifiedKFold(5,shuffle=True,random_state=args.seed)
for o_tr,te in outer.split(np.zeros(len(chem_idx)),y_bin):
    inner=MultilabelStratifiedShuffleSplit(n_splits=100,test_size=0.2,random_state=args.seed)
    for tr,va in inner.split(np.zeros(len(o_tr)),y_bin[o_tr]):
        if (y_bin[o_tr[va]].sum(0)>=args.min_pos).all():
            folds.append((torch.tensor(o_tr[tr]),
                          torch.tensor(o_tr[va]),
                          torch.tensor(te))); break

# ───────── 模型 ─────────
class HGT(torch.nn.Module):
    def __init__(self,in_dim,hid,meta,emb_dim,heads,chem_key):
        super().__init__()
        self.chem_key=chem_key; self.emb_dim=emb_dim
        self.gene_emb=torch.nn.Parameter(torch.zeros(emb_dim))
        self.path_emb=torch.nn.Parameter(torch.zeros(emb_dim))
        self.conv1=HGTConv(in_dim+emb_dim,hid,meta,heads=heads)
        self.conv2=HGTConv(hid,hid,meta,heads=heads)
        self.pred =torch.nn.Linear(hid,12)
    def _add(self,x_dict):
        out={}
        for k,x in x_dict.items():
            x=x.to(DEV); pad=torch.zeros(x.size(0),self.emb_dim,device=DEV)
            out[k]=torch.cat([x,pad],1)
        tail=slice(-self.emb_dim,None)
        out['gene'][:,tail]+=self.gene_emb
        out['path'][:,tail]+=self.path_emb
        return out
    def forward(self,x_dict):
        h=self.conv1(self._add(x_dict),edge_dict_gpu); h={k:torch.relu(v) for k,v in h.items()}
        h=self.conv2(h,edge_dict_gpu);                h={k:torch.relu(v) for k,v in h.items()}
        return self.pred(h[self.chem_key])

# ───────── 指标 ─────────
def per_receptor_metrics(y_true,y_prob):
    mask=(y_true!=-1)
    cols={'Receptor':RECEPTORS,'BAC':[],'F1':[],'AUC':[],'Toxic_Acc':[],'NonToxic_Acc':[]}
    for k in range(12):
        m=mask[:,k]
        if m.sum()==0:
            for key in cols.keys():
                if key!='Receptor': cols[key].append(np.nan)
            continue
        yt,yp=y_true[m,k],y_prob[m,k]; y_pred=(yp>=.5)
        cols['BAC' ].append(balanced_accuracy_score(yt,y_pred))
        cols['F1'  ].append(f1_score(yt,y_pred,zero_division=0))
        cols['AUC' ].append(roc_auc_score(yt,yp))
        tp=((yt==1)&(y_pred==1)).sum(); fn=((yt==1)&(y_pred==0)).sum()
        tn=((yt==0)&(y_pred==0)).sum(); fp=((yt==0)&(y_pred==1)).sum()
        cols['Toxic_Acc'   ].append(tp/(tp+fn+1e-9))
        cols['NonToxic_Acc'].append(tn/(tn+fp+1e-9))
    macro=(np.nanmean(cols['BAC']),np.nanmean(cols['F1']),np.nanmean(cols['AUC']))
    return macro,pd.DataFrame(cols)

# ───────── 训练 ─────────
fold_scores=[]; fold_aucs=[]; detail_all=[]
for f,(tr,va,te) in enumerate(folds):
    model=HGT(x_all.size(1),args.hidden,metadata,args.embed_dim,args.heads,chem_key).to(DEV)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    best_auc,best_state=-1,model.state_dict()

    tr_d,va_d,te_d=tr.to(DEV),va.to(DEV),te.to(DEV)
    tr_r,va_r,te_r=chem_loc[tr_d],chem_loc[va_d],chem_loc[te_d]

    # ---------- 计算 label-wise pos_weight (只看 train 集) ----------
    y_tr_cpu=y_full[tr_d].cpu().numpy()
    pos_cnt=(y_tr_cpu==1).sum(0); neg_cnt=(y_tr_cpu==0).sum(0)
    pos_weight=torch.tensor(neg_cnt/(pos_cnt+1e-6),device=DEV).clamp(max=10)  # ★ pos_weight

    criterion=torch.nn.BCEWithLogitsLoss(reduction='none',pos_weight=pos_weight)

    print(f"\n=== Fold {f} ===")
    for ep in tqdm(range(1,args.epochs+1),desc=f'Fold{f}',ncols=80):
        model.train(); opt.zero_grad()
        logits=model(x_dict_cpu)
        out_tr=logits.index_select(0,tr_r)
        y_tr  =y_full.index_select(0,tr_d)
        mask  =(y_tr!=-1).float()
        y_hat=y_tr.clone(); y_hat[y_hat==-1]=0
        loss_mat=criterion(out_tr,y_hat)
        loss   =(loss_mat*mask).sum()/mask.sum()      # ★ 使用加权 BCE
        loss.backward(); opt.step()

        if ep%50==0 or ep==args.epochs:
            model.eval()
            with torch.no_grad():
                val_p=torch.sigmoid(model(x_dict_cpu))
            val_auc=per_receptor_metrics(
                y_full[va_d].cpu().numpy(),
                val_p.index_select(0,va_r).cpu().numpy())[0][2]
            if val_auc>best_auc: best_auc,best_state=val_auc,model.state_dict()

    # 保存最佳
    torch.save(best_state,SAVE_DIR/f'fold{f}_best.pt')

    # 测试评估
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        te_p=torch.sigmoid(model(x_dict_cpu)).index_select(0,te_r).cpu().numpy()
    macro,detail=per_receptor_metrics(y_full[te_d].cpu().numpy(),te_p)
    fold_scores.append(macro); fold_aucs.append(macro[2]); detail_all.append(detail)

    detail.to_csv(SAVE_DIR/f'detail_fold{f}.csv',index=False)
    pd.DataFrame([[*macro]],columns=['BAC','F1','AUC']) \
        .to_csv(SAVE_DIR/f'metrics_fold{f}.csv',index=False)
    pd.DataFrame(te_p,columns=[f'label_{i}' for i in range(12)]) \
        .assign(chem_id=te.cpu().numpy()) \
        .to_csv(SAVE_DIR/f'fold{f}_pred.csv',index=False)

    print(f"Fold{f} TestAUC {macro[2]:.3f}")
    del model; torch.cuda.empty_cache(); gc.collect()

# ───────── 汇总 ─────────
summary_per=pd.concat(detail_all).groupby('Receptor').mean()
summary_per.to_csv(SAVE_DIR/'summary_per_receptor.csv')
macro_mean=np.array(fold_scores).mean(axis=0)
pd.DataFrame([macro_mean],columns=['BAC','F1','AUC']) \
    .to_csv(SAVE_DIR/'summary_macro.csv',index=False)

print('\n=== Per-receptor 5-fold Mean ==='); print(summary_per)
print('\n=== Macro 5-fold Mean ==='); print(dict(zip(['BAC','F1','AUC'],macro_mean)))

plt.figure(figsize=(6,4)); plt.bar(range(5),fold_aucs); plt.ylim(0,1)
plt.xticks(range(5),[f'Fold {i}' for i in range(5)])
plt.ylabel('Test AUC'); plt.title('HGT 5-fold AUC'); plt.tight_layout()
plt.savefig(SAVE_DIR/'auc_bar.png',dpi=300); plt.close()
print(f"[INFO] 所有结果已保存至 {SAVE_DIR.resolve()}")
