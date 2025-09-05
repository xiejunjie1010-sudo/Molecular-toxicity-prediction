import argparse
import random
import pandas as pd
import random
from rdkit import Chem
# from rdkit.Chem.rdmolops import GetAdjacencyMatrix
# from data import *
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, global_add_pool, Linear
from torch_geometric.loader import DataLoader

# from smote import random_over_sampling
import time
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from smote_fp import *

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_channels', type=int, default=512)
parser.add_argument('--num_features', type=int, default=52)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--num_classes', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--use_gdc', action='store', help='Use GDC')
args = parser.parse_args(args=[])

print("use_gdc", args.use_gdc)

start = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def auc_calculate(labels_auc,pred_auc):
    pos_cnt = labels_auc.sum()
    neg_cnt = len(pred_auc) - pos_cnt

    rank = np.argsort(pred_auc)
    ranklist = np.zeros(len(pred_auc))
    for i in range(len(pred_auc)):
        if labels_auc[rank[i]] == 1:
            ranklist[rank[i]] = i + 1
    auc = (ranklist.sum() - pos_cnt * (pos_cnt + 1) / 2) / (pos_cnt * neg_cnt)
    return auc

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, cached=False,
                             normalize=not args.use_gdc)
        self.conv2 = GATConv(hidden_channels, hidden_channels * 2, cached=False,
                             normalize=not args.use_gdc)
        # self.fc1 = Linear(hidden_channels*2, hidden_channels)  # //表示取整，即转成整型
        self.fc2 = Linear(hidden_channels * 2, out_channels)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data):
        x, edge_index,edge_attr, batch = data.x, data.edge_index,data.edge_attr, data.batch
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = global_add_pool(x, batch)
        # x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def model_train(train_loader):
    epoch_train_loss = 0
    model.train()
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        out = torch.squeeze(out)
        # print(i, out.size(), data.y.size())
        loss = F.binary_cross_entropy(out, data.y)
        epoch_train_loss += loss.item()
        loss.backward()
        optimizer.step()
    epoch_train_loss /= len(train_loader)
    return float(epoch_train_loss)


@torch.no_grad()
def model_test(test_loader):
    epoch_test_loss = 0
    model.eval()
    for i, data in enumerate(test_loader):
        data = data.to(device)

        out = model(data)
        out = torch.squeeze(out)

        loss = F.binary_cross_entropy(out, data.y)
        epoch_test_loss += loss.item()

    epoch_test_loss /= len(test_loader)
    return float(epoch_test_loss)


@torch.no_grad()
def model_pred(data_loader):
    model.eval()
    y_test = np.array([])
    y_test_pred = np.array([])
    for i, data in enumerate(data_loader):
        data = data.to(device)
        pred = model(data)

        y_test = np.append(y_test, data.y.cpu().numpy())
        y_test_pred = np.append(y_test_pred, pred.cpu().numpy())
    return y_test, y_test_pred

def k_fold_cross_validation_split(datasets, folds=5):
    basket_split_data = list()
    fold_size = int(len(datasets)/folds)
    idx = list(range(len(datasets)))
    while len(basket_split_data) < folds:
        basket_random_fold = list()
        for i in range(fold_size):
            random_choose_index = np.random.choice(idx)
            basket_random_fold.append(datasets[random_choose_index])
            idx.remove(random_choose_index)
        basket_split_data.append(basket_random_fold)
    train_data = list()
    test_data = list()
    for i in range(folds):
        test_data.append(basket_split_data[i])
        temp = list(range(folds))
        temp.remove(i)
        train_temp = list()
        for j in temp:
            train_temp.extend(basket_split_data[j])
        train_data.append(train_temp)
    return train_data, test_data


def accuracy(predict_values, actual):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == 0.0 and predict_values[i] < 0.5 or actual[i] == 1.0 and predict_values[i] >= 0.5:
            correct += 1
    return correct / float(len(actual))


raw_file_names = ["nr-ahr.smiles", "nr-ar.smiles", "nr-ar-lbd.smiles", "nr-aromatase.smiles", "nr-er.smiles",
                  "nr-er-lbd.smiles", "nr-ppar-gamma.smiles",
                  "sr-are.smiles", "sr-atad5.smiles", "sr-hse.smiles", "sr-mmp.smiles", "sr-p53.smiles"]

res_acc_auc_none_k = list()
for filename in raw_file_names:
    mean_aucs = list()
    mean_accs = list()
    datasets = pd.read_csv('../data/raw/' + filename, sep='\t', header=None, names=["Smiles", "Sample ID", "Label"],
                           encoding='utf-8').reset_index()
    filename = filename.split('.')[0]
    num = 0
    for smile in datasets['Smiles']:
        mol = Chem.MolFromSmiles(smile)
        if mol:
            num += 1
    datasets = [torch.load(f'../data/processed/{filename}_data{i}.pt') for i in range(num)]

    # # 随机划分训练集和测试集
    # index_list = list(range(num))
    # train_data = list()
    # test_data = list()
    #
    # random.shuffle(index_list)
    # for index, j in enumerate(index_list):
    #     if index <= len(index_list) * 0.8:
    #         train_data.append(datasets[j])
    #     else:
    #         test_data.append(datasets[j])
    #
    # print(f'数据集中包含的图的数量{len(datasets)}\ntrain_data:{len(train_data)} test_data:{len(test_data)}')

    # 五折交叉检验
    folds = 5
    # 定义空列表用于保存每次的训练结果
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    score = []
    train_data, test_data = k_fold_cross_validation_split(datasets, folds)

    for i in range(folds):

        if args.use_gdc:
            transform = T.GDC(
                self_loop_weight=1,
                normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128, dim=0),
                exact=True,
            )
            for data in train_data[i]:
                data.edge_attr = None
                data = transform(data)

        smote = SmoteXAttrAvg(k=5)  # Expected n_neighbors <= n_samples
        train_data[i] = smote.fit(train_data[i])
        print(f'第{i}折中包含的图的数量:{len(train_data[i]) + len(test_data[i])}\ntrain_data[{i}]:{len(train_data[i])} test_data[{i}]:{len(test_data[i])}')

        model = GAT(args.num_features, args.hidden_channels, args.num_classes)
        seed_torch(105)
        model = model.to(device)
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=5e-4),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=args.lr)  # Only perform weight-decay on first convolution.

        train_loader = DataLoader(
            dataset=train_data[i],
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True
        )
        # val_loader = DataLoader(
        #     dataset=val_data,
        #     batch_size=args.batch_size,
        #     shuffle=False,
        #     drop_last=True
        # )
        test_loader = DataLoader(
            dataset=test_data[i],
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True
        )

        for epoch in range(1, args.epochs + 1):
            loss = model_train(train_loader)
            y_test, y_test_pred = model_pred(test_loader)
            acc = accuracy(y_test_pred, y_test)

            print(f'Epoch: {epoch:04d}, Train_Loss: {loss:.3f} Test_Acc: {acc:.3f}')

        # print(f'测试集最高精度：{test_acc_best:.3f}, epoch:{test_epoch_best:.3f}')
        # res_acc_epoch_none.append([filename, test_acc_best, test_epoch_best])

        # y_train, y_train_pred = model_pred(train_loader)
        y_test, y_test_pred = model_pred(test_loader)

        acc = accuracy(y_test_pred, y_test)
        score.append(acc)

        # 计算fpr(假阳性率),tpr(真阳性率),thresholds(阈值)[绘制ROC曲线要用到这几个值]
        fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
        # interp:插值 把结果添加到tprs列表中
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        # 计算auc
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

     # 画对角线
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc), lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', label=r"$\pm$ 1 std. dev.", alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc='lower right')
    plt.savefig('./save/smote_x_attr_more_avg_5_gat/ROC_%s.jpg' % filename)
    # plt.show()
    plt.close()
    # mean_aucs.append(mean_auc)
    # mean_accs.append(np.mean(score))
    print("acc", filename, ":", np.mean(score))
    np.save('./save/smote_x_attr_more_avg_5_gat/auc_%s.npy' % filename, aucs)
    np.save('./save/smote_x_attr_more_avg_5_gat/acc_%s.npy' % filename, score)

# 计算模型运行的时间
end = time.time()
print(end - start, "s")
