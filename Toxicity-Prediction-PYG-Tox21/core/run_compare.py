#!/usr/bin/env python
"""
core/run_compare.py
===================
统一比较 5 种模型:
  • GCN / GAT / GraphSAGE     （化合物子环图）
  • Random-Forest             （多标签）
  • R-GCN                     （fp_only: 自环；fp_kg: 多关系 + gene/pathway 可训练嵌入）

特征方案:
  fp_only  ->  仅指纹
  fp_kg    ->  指纹 + KG 嵌入 (node_features.npy)

结果:
  logs/{model}/{feature}_fold{fold}.csv
"""

# ------------------------------------------------------------------
# 通用依赖
# ------------------------------------------------------------------
import argparse, json, pickle, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, confusion_matrix  # ★ 新增
)
from tqdm import tqdm
from utils_metrics import balanced_accuracy, toxic_nontoxic_acc
from models.rf_wrapper import RF

# ------------------------------------------------------------------
# 路径常量
# ------------------------------------------------------------------
ROOT  = Path(__file__).resolve().parent.parent
KGDIR = ROOT / "data" / "KG"
RGDIR = KGDIR / "r-gcn"

# ------------------------------------------------------------------
# 1. 数据加载
# ------------------------------------------------------------------
compound   = pd.read_csv(KGDIR / "compound_master.csv")
FP_COLS    = [c for c in compound.columns if c.startswith("fp_")]
labels_mat = np.load(KGDIR / "label_matrix.npy")        # chem × 12
fold_masks = pickle.load(open(KGDIR / "fold_masks.pkl", "rb"))

# -- CID↔node_id (仅化合物) --
with open(RGDIR / "node_id_map.json") as f:
    cid2nid = {int(k): v for k, v in json.load(f).items() if k.isdigit()}

cids      = compound["cid"].tolist()
nid_order = [cid2nid[c] for c in cids]

# -- 特征矩阵 --
node_feat = np.load(RGDIR / "node_features.npy")
feat_fpkg = torch.tensor(node_feat[nid_order], dtype=torch.float32)
feat_fp   = torch.tensor(compound[FP_COLS].values, dtype=torch.float32)

# -- R-GCN专用 --
edge_index = torch.from_numpy(np.load(RGDIR / "edge_index.npy")).long()
edge_type  = torch.from_numpy(np.load(RGDIR / "edge_type.npy")).long()
NUM_REL    = edge_type.max().item() + 1
gene_idx   = torch.from_numpy(np.load(RGDIR / "gene_idx.npy")).long()
path_idx   = torch.from_numpy(np.load(RGDIR / "pathway_idx.npy")).long()

# -- 自环子图 --
N_CHEM   = len(compound)
loop_edge = torch.arange(N_CHEM).repeat(2, 1)                # [2, N]
loop_type = torch.zeros(loop_edge.size(1), dtype=torch.long) # 单关系 id=0

RECEPTORS = compound.columns[-12:]

# ------------------------------------------------------------------
# 2. 评价函数（自动忽略单类 AUC 错误）
# ------------------------------------------------------------------
def evaluate(y_true, prob):
    """
    返回 DataFrame(index=receptor,
                  columns=[AUC, PR_AUC, F1, BAC, ToxicAcc, NontoxAcc])
    • AUC / PR-AUC 遇单一类别置 NaN
    • ToxicAcc / NontoxAcc 基于 confusion_matrix 计算，避免按位运算
    """
    results = {}
    for i, rec in enumerate(RECEPTORS):
        mask = y_true[:, i] != -1
        if mask.sum() == 0:
            continue

        yt = y_true[mask, i].astype(int)          # → 0 / 1
        yp = prob[mask, i]
        pred = (yp > 0.5).astype(int)

        # --- AUC / PR-AUC ---
        if yt.max() == yt.min():                  # 仅一个类别
            auc_val, pr_auc = np.nan, np.nan
        else:
            auc_val  = roc_auc_score(yt, yp)
            pr_auc   = average_precision_score(yt, yp)

        # --- 其余指标 ---
        f1  = f1_score(yt, pred, zero_division=0)
        bac = balanced_accuracy(yt, pred)

        tn, fp, fn, tp = confusion_matrix(
            yt, pred, labels=[0, 1]
        ).ravel()
        tox_acc, nontox_acc = toxic_nontoxic_acc(tp, fp, tn, fn)

        results[rec] = dict(
            AUC       = auc_val,
            PR_AUC    = pr_auc,
            F1        = f1,
            BAC       = bac,
            ToxicAcc  = tox_acc,
            NontoxAcc = nontox_acc
        )

    return pd.DataFrame(results).T

# ------------------------------------------------------------------
# 3. 基本 GNN (GCN / GAT / GraphSAGE) —— 自环图
# ------------------------------------------------------------------
from torch_geometric.data import Data
def pyg_train(model_cls, features, masks, epochs=300, lr=1e-3):
    n = features.size(0)
    data = Data(
        x=features,
        edge_index=loop_edge,
        y=torch.tensor(labels_mat, dtype=torch.float32)
    )
    data.batch = torch.arange(n)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = model_cls(in_dim=features.size(1)).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    lossf  = torch.nn.BCEWithLogitsLoss(reduction='none')

    tr, _, te = [torch.tensor(m, dtype=torch.long) for m in masks]
    for _ in tqdm(range(epochs), desc="train", leave=False):
        model.train(); opt.zero_grad()
        out  = model(data.to(device))
        loss = lossf(out[tr], data.y[tr]); loss = loss[data.y[tr]!=-1].mean()
        loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(data.to(device))).cpu().numpy()
    return labels_mat[te], prob[te]

# ------------------------------------------------------------------
# 4. Random-Forest
# ------------------------------------------------------------------
def run_rf(X, masks):
    tr, _, te = masks
    rf = RF().fit(X[tr], labels_mat[tr])
    prob_list = rf.predict_proba(X[te])
    prob = np.stack([p[:, 1] for p in prob_list], 1)
    return labels_mat[te], prob

# ------------------------------------------------------------------
# 5. R-GCN (带开关 use_kg)
# ------------------------------------------------------------------
class RGCN(torch.nn.Module):
    def __init__(self, in_dim, h_dim, num_rel, emb_dim,
                 n_gene, n_path):
        super().__init__()
        self.emb_dim = emb_dim
        from torch_geometric.nn import RGCNConv
        if emb_dim:
            self.gene_emb = torch.nn.Embedding(n_gene, emb_dim)
            self.path_emb = torch.nn.Embedding(n_path, emb_dim)
        else:
            self.register_parameter("gene_emb", None)
            self.register_parameter("path_emb", None)
        self.conv1 = RGCNConv(in_dim + emb_dim, h_dim, num_rel, num_bases=8)
        self.conv2 = RGCNConv(h_dim, h_dim, num_rel, num_bases=8)
        self.lin   = torch.nn.Linear(h_dim, 12)

    def forward(self, x, ei, et, gene_idx, path_idx):
        if self.emb_dim:                       # 加可训练嵌入
            pad = torch.zeros(x.size(0), self.emb_dim, device=x.device)
            x   = torch.cat([x, pad], 1)
            tail = slice(-self.emb_dim, None)
            x[gene_idx,  tail] += self.gene_emb.weight
            x[path_idx, tail]  += self.path_emb.weight
        h = torch.relu(self.conv1(x, ei, et))
        h = torch.relu(self.conv2(h, ei, et))
        return self.lin(h)

def train_rgcn(masks, use_kg: bool,
               epochs, hidden=512, lr=1e-3, emb_dim=64):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if use_kg:                               # 指纹+KG + 全图
        x_dev, ei, et = feat_fpkg.to(device), edge_index.to(device), edge_type.to(device)
        g_idx, p_idx  = gene_idx.to(device), path_idx.to(device)
        emb, n_rel    = emb_dim, NUM_REL
    else:                                    # 纯指纹 + 自环
        x_dev, ei, et = feat_fp.to(device), loop_edge.to(device), loop_type.to(device)
        g_idx = p_idx = torch.tensor([], dtype=torch.long, device=device)
        emb, n_rel    = 0, 1

    model = RGCN(x_dev.size(1), hidden, n_rel, emb,
                 len(gene_idx), len(path_idx)).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    tr, val, te = [torch.tensor(m, dtype=torch.long, device=device) for m in masks]
    lab = torch.tensor(labels_mat, dtype=torch.float32).to(device)

    pos = (lab == 1).float(); neg = (lab == 0).float()
    pos_w = (neg.sum(0) / (pos.sum(0) + 1e-6)).clamp(max=10).to(device)
    lossf = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_w)

    best_auc, best_state = -1, None
    for ep in tqdm(range(1, epochs + 1), desc="RGCN-train", leave=False):
        model.train(); opt.zero_grad()
        out  = model(x_dev, ei, et, g_idx, p_idx)
        loss = lossf(out[tr], lab[tr]); loss = loss[lab[tr] != -1].mean()
        loss.backward(); opt.step()

        if ep % 100 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                val_out = model(x_dev, ei, et, g_idx, p_idx)
            # 若单类别会报错，用 try/except 跳过
            try:
                val_auc = roc_auc_score(
                    lab[val].cpu().numpy(),
                    torch.sigmoid(val_out[val]).cpu().numpy(),
                    average='macro', multi_class='raise')
            except ValueError:
                val_auc = -1
            if val_auc > best_auc:
                best_auc, best_state = val_auc, model.state_dict()

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(x_dev, ei, et, g_idx, p_idx))
    return lab[te].cpu().numpy(), prob[te].cpu().numpy()

# ------------------------------------------------------------------
# 6. CLI 主程序
# ------------------------------------------------------------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--model",   choices=["GCN","GAT","GraphSAGE","RF","RGCN"])
    pa.add_argument("--feature", choices=["fp_only","fp_kg"])
    pa.add_argument("--fold",    type=int, default=0)
    pa.add_argument("--epochs",  type=int, default=300)
    args = pa.parse_args()

    masks = fold_masks[args.fold]

    # ----- Random-Forest -----
    if args.model == "RF":
        feat = feat_fp if args.feature == "fp_only" else feat_fpkg
        y_true, prob = run_rf(feat.numpy(), masks)

    # ----- 基本 GNN -------
    elif args.model in ["GCN","GAT","GraphSAGE"]:
        feat = feat_fp if args.feature == "fp_only" else feat_fpkg
        if args.model == "GCN":
            from models.gcn_wrapper import Net as M
        elif args.model == "GAT":
            from models.gat_wrapper import Net as M
        else:
            from models.sage_wrapper import Net as M
        y_true, prob = pyg_train(M, feat, masks, epochs=args.epochs)

    # ----- R-GCN ---------
    elif args.model == "RGCN":
        use_kg = (args.feature == "fp_kg")
        y_true, prob = train_rgcn(
            masks, use_kg=use_kg,
            epochs=max(args.epochs, 800 if use_kg else args.epochs),
            hidden=512, lr=1e-3, emb_dim=64
        )

    else:
        raise ValueError("未知模型")

    # ----- 保存 CSV -------
    df = evaluate(y_true, prob)
    out_dir = ROOT / "logs" / args.model.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.feature}_fold{args.fold}.csv"
    df.to_csv(out_path)

    print(f"\n✅  保存 → {out_path.relative_to(ROOT)}")
    print("平均指标:\n", df.mean())
