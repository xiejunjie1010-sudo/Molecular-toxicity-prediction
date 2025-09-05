"""
GAT + SMOTE-XAttrAvg 5-fold CV
│─ 指标：BAC / F1 / AUC
│─ 结果：./save/smote_x_attr_more_avg_5_gat/metrics_<task>.csv, ROC_<task>.jpg
│─ 进度：tqdm
"""

import argparse, random, time, numpy as np, pandas as pd
from pathlib import Path
from rdkit import Chem
import torch, torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_add_pool, Linear
from sklearn.metrics import (
    balanced_accuracy_score, f1_score,
    roc_curve, roc_auc_score, auc,
)
from matplotlib import pyplot as plt
from tqdm import tqdm
from smote_fp import SmoteXAttrAvg     # 过采样

# ------------------------------------------------------------------ #
# 0. 解析参数
# ------------------------------------------------------------------ #
parser = argparse.ArgumentParser()
parser.add_argument('--hidden_channels', type=int, default=512)
parser.add_argument('--num_features',    type=int, default=52)
parser.add_argument('--lr',              type=float, default=1e-3)
parser.add_argument('--epochs',          type=int, default=1000)
parser.add_argument('--num_classes',     type=int, default=1)
parser.add_argument('--batch_size',      type=int, default=64)
parser.add_argument('--use_gdc',         action='store_true', help='Use GDC')
args = parser.parse_args(args=[])

device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_dir = Path('./save/smote_x_attr_more_avg_5_gat_bac')
save_dir.mkdir(parents=True, exist_ok=True)
print('[Device]', device, '| use_gdc =', args.use_gdc)

# ------------------------------------------------------------------ #
# 1. 工具函数
# ------------------------------------------------------------------ #
def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def eval_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    bac  = balanced_accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    aucv = roc_auc_score(y_true, y_prob)
    return bac, f1, aucv

def k_fold_split(ds, k=5):
    idx = list(range(len(ds))); random.shuffle(idx)
    fold_size = len(ds)//k
    splits = [idx[i*fold_size:(i+1)*fold_size] for i in range(k)]
    trains, tests = [], []
    for i in range(k):
        tests.append([ds[j] for j in splits[i]])
        trains.append([ds[j] for f in range(k) if f!=i for j in splits[f]])
    return trains, tests

@torch.no_grad()
def predict(loader, model):
    model.eval(); y_t, y_p = [], []
    for data in loader:
        data = data.to(device)
        prob = torch.squeeze(model(data)).cpu().numpy()
        y_t.extend(data.y.cpu().numpy())
        y_p.extend(prob)
    return np.array(y_t), np.array(y_p)

# ------------------------------------------------------------------ #
# 2. 模型
# ------------------------------------------------------------------ #
class GAT(torch.nn.Module):
    def __init__(self, in_c, hid, out_c):
        super().__init__()
        self.conv1 = GATConv(in_c, hid, heads=1)
        self.conv2 = GATConv(hid,  hid*2, heads=1)
        self.fc    = Linear(hid*2, out_c)
        self.sig   = torch.nn.Sigmoid()
    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, ei).relu()
        x = self.conv2(x, ei).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = global_add_pool(x, batch)
        return self.sig(self.fc(x))

def train_epoch(loader, model, opt):
    model.train(); ep_loss = 0
    for data in loader:
        data = data.to(device)
        opt.zero_grad()
        out = torch.squeeze(model(data))
        loss = F.binary_cross_entropy(out, data.y)
        loss.backward(); opt.step()
        ep_loss += loss.item()
    return ep_loss / len(loader)

# ------------------------------------------------------------------ #
# 3. 主流程
# ------------------------------------------------------------------ #
start_time = time.time()
tasks = [
    "nr-ahr","nr-ar","nr-ar-lbd","nr-aromatase","nr-er",
    "nr-er-lbd","nr-ppar-gamma",
    "sr-are","sr-atad5","sr-hse","sr-mmp","sr-p53"
]

for task in tasks:
    print(f'\n### Task: {task} ###')
    df = pd.read_csv(f'../data/raw/{task}.smiles', sep='\t',
                     header=None, names=['Smiles','ID','Label'])
    num = sum(bool(Chem.MolFromSmiles(s)) for s in df['Smiles'])
    dataset = [torch.load(f'../data/processed/{task}_data{i}.pt')
               for i in range(num)]

    train_sets, test_sets = k_fold_split(dataset, k=5)
    mean_fpr = np.linspace(0,1,100)
    tprs, bac_ls, f1_ls, auc_ls = [], [], [], []

    for fold in range(5):
        print(f'  Fold {fold+1}/5 ➜ train {len(train_sets[fold])}  test {len(test_sets[fold])}')
        # —— 可选 GDC
        if args.use_gdc:
            gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                        normalization_out='col',
                        diffusion_kwargs=dict(method='ppr', alpha=0.05),
                        sparsification_kwargs=dict(method='topk', k=128, dim=0),
                        exact=True)
            for idx_d, d in enumerate(train_sets[fold]):
                d.edge_attr = None
                train_sets[fold][idx_d] = gdc(d)

        # —— 过采样
        smote = SmoteXAttrAvg(k=5)
        train_sets[fold] = smote.fit(train_sets[fold])

        model = GAT(args.num_features, args.hidden_channels, args.num_classes).to(device)
        seed_all(105+fold)
        optimizer = torch.optim.Adam(
            [{'params': model.conv1.parameters(), 'weight_decay':5e-4},
             {'params': model.conv2.parameters(), 'weight_decay':0}],
            lr=args.lr)

        train_loader = DataLoader(train_sets[fold], batch_size=args.batch_size,
                                  shuffle=True, drop_last=True)
        test_loader  = DataLoader(test_sets[fold],  batch_size=args.batch_size,
                                  shuffle=False, drop_last=True)

        # —— 训练
        for ep in tqdm(range(1, args.epochs+1),
                       desc=f'    Epochs (fold {fold+1})', leave=False):
            loss = train_epoch(train_loader, model, optimizer)
            if ep % 100 == 0 or ep in (1, args.epochs):
                y_t, y_p = predict(test_loader, model)
                bac, f1, aucv = eval_metrics(y_t, y_p)
                tqdm.write(f'      Epoch {ep:04d}  Loss {loss:.3f}  '
                           f'BAC {bac:.3f}  F1 {f1:.3f}  AUC {aucv:.3f}')

        # —— 折评估
        y_t, y_p = predict(test_loader, model)
        bac, f1, aucv = eval_metrics(y_t, y_p)
        bac_ls.append(bac); f1_ls.append(f1); auc_ls.append(aucv)

        fpr, tpr, _ = roc_curve(y_t, y_p)
        interp = np.interp(mean_fpr, fpr, tpr); interp[0] = 0.0
        tprs.append(interp)
        plt.plot(fpr, tpr, lw=1, alpha=.3,
                 label=f'Fold {fold+1} (AUC={aucv:.2f})')

    # —— 平均 ROC 图
    plt.plot([0,1],[0,1],'--',lw=2,color='r',alpha=.8,label='Chance')
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr); std_auc = np.std(auc_ls)
    plt.plot(mean_fpr, mean_tpr, color='b',
             lw=2, alpha=.8,
             label=f'Mean ROC (AUC={mean_auc:.2f}±{std_auc:.2f})')
    std_tpr = np.std(tprs, axis=0)
    plt.fill_between(mean_fpr, np.maximum(mean_tpr-std_tpr,0),
                     np.minimum(mean_tpr+std_tpr,1),
                     color='grey', alpha=.2, label='±1 std')
    plt.xlim([-0.05,1.05]); plt.ylim([-0.05,1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC — {task}'); plt.legend(loc='lower right')
    plt.savefig(save_dir / f'ROC_{task}.jpg'); plt.close()

    # —— 保存 CSV
    pd.DataFrame({
        'fold': np.arange(1,6),
        'BAC':  bac_ls,
        'F1':   f1_ls,
        'AUC':  auc_ls
    }).to_csv(save_dir / f'metrics_{task}.csv',
              index=False, float_format='%.6f')

    print(f'  >>> 5-fold mean BAC {np.mean(bac_ls):.3f}  '
          f'F1 {np.mean(f1_ls):.3f}  AUC {np.mean(auc_ls):.3f}')

print(f'\n[Elapsed] {time.time() - start_time:.1f}s')
