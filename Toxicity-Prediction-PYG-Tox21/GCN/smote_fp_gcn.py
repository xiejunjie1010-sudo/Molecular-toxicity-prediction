import random

from rdkit.Chem.Draw import SimilarityMaps
from rdkit.SimDivFilters import MaxMinPicker
from sklearn.neighbors import NearestNeighbors
import numpy as np
from torch_geometric.data import Dataset, Data
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
import time

class SmoteXAttrGCN:
    """
    SMOTE过采样算法.


    Parameters:
    -----------
    k: int
        选取的近邻数目.
    sampling_rate: int
        采样倍数, attention sampling_rate < k.
    newindex: int
        生成的新样本(合成样本)的索引号.
    """

    def __init__(self, k=5):
        self.sampling_rate = 1
        self.k = k

    def fit(self, datasets):
        self.datasets_smote = datasets  # 浅复制

        positive = list()
        negative = list()

        for data in self.datasets_smote:
            if data.y == 1:
                positive.append(data)
            else:
                negative.append(data)

        len_negative = len(negative)
        len_positive = len(positive)

        print("smote_x_attr_gcn:k=%d nontox_idx:%d tox_idx:%d" % (self.k, len_negative, len_positive))

        if abs(len_positive - len_negative) <= (min(len_positive, len_negative) * 0.5):
            return self.datasets_smote
        elif len_positive < len_negative:
            self.sampling_rate = 1 if (((len_negative - len_positive) // len_positive) == 0) else (
                        (len_negative - len_positive) // len_positive)
            print("smote_x_attr_gcn:sampling_rate=%d nontox_idx:%d tox_idx:%d" % (
            self.sampling_rate, len_negative, len_positive * (self.sampling_rate + 1)))
            self.get_samples(positive)
            return self.datasets_smote
        else:
            self.sampling_rate = 1 if (((len_positive - len_negative) // len_negative) == 0) else (
                    (len_positive - len_negative) // len_negative)
            print("smote_x_attr_gcn:sampling_rate=%d nontox_idx:%d tox_idx:%d" % (
            self.sampling_rate, len_negative * (self.sampling_rate + 1), len_positive))
            self.get_samples(negative)
            return self.datasets_smote

    def get_feature(self, samples):
        feature = np.zeros(shape=(len(samples), 52 + 10))
        for i, data in enumerate(samples):
            temp = list()
            x = data.x.sum(axis=0)
            for e in x:
                if e.item():
                    temp.append(e.item() / data.x.shape[0])
                else:
                    temp.append(e.item())
            x = data.edge_attr.sum(axis=0)
            for e in x:
                if e.item():
                    temp.append(e.item() / data.edge_attr.shape[0])
                else:
                    temp.append(e.item())
            feature[i] = np.array(temp)
        return feature

    def get_samples(self, minority_samples):

        feature = self.get_feature(minority_samples)
        # 找出某一类样本集(数据集X)中的每一个样本在数据集X中的k个近邻
        knn = NearestNeighbors(n_neighbors=self.k).fit(feature)

        for i in range(len(minority_samples)):

            k_neighbors = knn.kneighbors(feature[i].reshape(1, -1), return_distance=False)[0]
            # 对某一类样本集(minority class samples)中每个样本, 分别根据其k个近邻生成

            # sampling_rate个新的样本
            for j in range(self.sampling_rate):
                # 从k个近邻里面随机选择一个近邻
                neighbor = np.random.choice(k_neighbors)

                # diff_x
                l1 = minority_samples[neighbor].x.shape[0]
                l2 = minority_samples[i].x.shape[0]
                # 计算样本X[i]与刚刚选择的近邻的差
                if l1 >= l2:
                    diff_x = minority_samples[neighbor].x[:l2, :] - minority_samples[i].x
                else:
                    zero_x = torch.zeros(l2 - l1, minority_samples[i].x.shape[1])
                    diff_x = torch.cat(((minority_samples[neighbor].x - minority_samples[i].x[:l1, :]), zero_x), dim=0)

                # diff_attr
                l1 = minority_samples[neighbor].edge_attr.shape[0]
                l2 = minority_samples[i].edge_attr.shape[0]
                # 计算样本edge_attr[i]与刚刚选择的近邻的差
                if l1 >= l2:
                    diff_attr = minority_samples[neighbor].edge_attr[:l2, :] - minority_samples[i].edge_attr
                else:
                    zero_attr = torch.zeros(l2 - l1, minority_samples[i].edge_attr.shape[1])
                    diff_attr = torch.cat(
                        ((minority_samples[neighbor].edge_attr - minority_samples[i].edge_attr[:l1, :]), zero_attr),
                        dim=0)

                # 生成新的数据
                self.datasets_smote.append(Data(x=(minority_samples[i].x + random.random() * diff_x),
                                                edge_index=minority_samples[i].edge_index,
                                                edge_attr=(minority_samples[i].edge_attr + random.random() * diff_attr),
                                                y=minority_samples[i].y))

class SmoteXAttrAvg:
    """
    SMOTE过采样算法.


    Parameters:
    -----------
    k: int
        选取的近邻数目.
    sampling_rate: int
        采样倍数, attention sampling_rate < k.
    newindex: int
        生成的新样本(合成样本)的索引号.
    """

    def __init__(self, k=5):
        self.sampling_rate = 1
        self.k = k

    def fit(self, datasets):
        self.datasets_smote = datasets  # 浅复制

        positive = list()
        negative = list()

        for data in self.datasets_smote:
            if data.y == 1:
                positive.append(data)
            else:
                negative.append(data)

        len_negative = len(negative)
        len_positive = len(positive)

        print("smote_x_attr_avg:k=%d nontox_idx:%d tox_idx:%d" % (self.k, len_negative, len_positive))

        if len_positive == len_negative:
            return self.datasets_smote
        elif len_positive < len_negative:
            self.more = len_negative
            self.less = len_positive
            self.sampling_rate = 1 + (len_negative - len_positive) // len_positive
            print("smote_x_attr_avg:sampling_rate=%d nontox_idx:%d tox_idx:%d" % (
            self.sampling_rate, len_negative, len_positive * (self.sampling_rate + 1)))
            self.get_samples(positive)
            return self.datasets_smote
        else:
            self.more = len_positive
            self.less = len_negative
            self.sampling_rate = 1 + (len_positive - len_negative) // len_negative
            print("smote_x_attr_avg:sampling_rate=%d nontox_idx:%d tox_idx:%d" % (
            self.sampling_rate, len_negative * (self.sampling_rate + 1), len_positive))
            self.get_samples(negative)
            return self.datasets_smote

    def get_feature(self, samples):
        feature = np.zeros(shape=(len(samples), 52 + 10))
        for i, data in enumerate(samples):
            temp = list()
            x = data.x.sum(axis=0)
            for e in x:
                if e.item():
                    temp.append(e.item() / data.x.shape[0])
                else:
                    temp.append(e.item())
            x = data.edge_attr.sum(axis=0)
            for e in x:
                if e.item():
                    temp.append(e.item() / data.edge_attr.shape[0])
                else:
                    temp.append(e.item())
            feature[i] = np.array(temp)
        return feature

    def get_samples(self, minority_samples):

        feature = self.get_feature(minority_samples)
        # 找出某一类样本集(数据集X)中的每一个样本在数据集X中的k个近邻
        knn = NearestNeighbors(n_neighbors=self.k).fit(feature)

        for i in range(len(minority_samples)):

            k_neighbors = knn.kneighbors(feature[i].reshape(1, -1), return_distance=False)[0]
            # 对某一类样本集(minority class samples)中每个样本, 分别根据其k个近邻生成

            # sampling_rate个新的样本
            for j in range(self.sampling_rate):
                # 从k个近邻里面随机选择一个近邻
                neighbor = np.random.choice(k_neighbors)

                # diff_x
                temp = list()
                x = minority_samples[neighbor].x.sum(axis=0)
                for e in x:
                    if e.item():
                        temp.append(e.item() / minority_samples[neighbor].x.shape[0])
                    else:
                        temp.append(e.item())
                x_avg = torch.Tensor([temp for k in range(minority_samples[i].x.shape[0])])
                # print("x_avg:",x_avg.shape,minority_samples[i].x.shape)
                # 计算样本X[i]与刚刚选择的近邻的差
                diff_x = x_avg - minority_samples[i].x

                # diff_attr
                temp = list()
                edge_attr = minority_samples[neighbor].edge_attr.sum(axis=0)
                for e in edge_attr:
                    if e.item():
                        temp.append(e.item() / minority_samples[neighbor].edge_attr.shape[0])
                    else:
                        temp.append(e.item())
                if minority_samples[i].edge_attr.shape[0] != 0:
                    attr_avg = torch.Tensor([temp for k in range(minority_samples[i].edge_attr.shape[0])])
                else:
                    attr_avg = torch.zeros(0, 10, dtype=torch.float)
                # print("attr_avg:",attr_avg.shape,minority_samples[i].edge_attr.shape)
                # 计算样本X[i]与刚刚选择的近邻的差
                diff_attr = attr_avg - minority_samples[i].edge_attr

                # 生成新的数据
                self.datasets_smote.append(Data(x=(minority_samples[i].x + random.random() * diff_x),
                                                edge_index=minority_samples[i].edge_index,
                                                edge_attr=(minority_samples[i].edge_attr + random.random() * diff_attr),
                                                y=minority_samples[i].y))

class SmoteXAttrMaxMin:
    """
    SMOTE过采样算法.


    Parameters:
    -----------
    k: int
        选取的近邻数目.
    sampling_rate: int
        采样倍数, attention sampling_rate < k.
    newindex: int
        生成的新样本(合成样本)的索引号.
    """

    def __init__(self, k=5):
        self.sampling_rate = 1
        self.k = k

    def fit(self, datasets):
        self.datasets_smote = datasets  # 浅复制

        self.smote_samples = list()

        positive = list()
        negative = list()

        for data in self.datasets_smote:
            if data.y == 1:
                positive.append(data)
            else:
                negative.append(data)

        len_negative = len(negative)
        len_positive = len(positive)

        print("smote_x_attr_MaxMin_gcn:k=%d nontox_idx:%d tox_idx:%d" % (self.k, len_negative, len_positive))

        if len_positive == len_negative:
            return self.datasets_smote
        elif len_positive < len_negative:
            self.more = len_negative
            self.less = len_positive
            self.sampling_rate = 1 + (len_negative - len_positive) // len_positive
            print("smote_x_attr_MaxMin_gcn:sampling_rate=%d nontox_idx:%d tox_idx:%d" % (
            self.sampling_rate, len_negative, len_positive * (self.sampling_rate + 1)))
            self.get_samples(positive)
            return self.datasets_smote
        else:
            self.more = len_positive
            self.less = len_negative
            self.sampling_rate = 1 + (len_positive - len_negative) // len_negative
            print("smote_x_attr_MaxMin_gcn:sampling_rate=%d nontox_idx:%d tox_idx:%d" % (
            self.sampling_rate, len_negative * (self.sampling_rate + 1), len_positive))
            self.get_samples(negative)
            return self.datasets_smote

    def get_feature(self, samples):
        feature = np.zeros(shape=(len(samples), 52 + 10))
        for i, data in enumerate(samples):
            temp = list()
            x = data.x.sum(axis=0)
            for e in x:
                if e.item():
                    temp.append(e.item() / data.x.shape[0])
                else:
                    temp.append(e.item())
            x = data.edge_attr.sum(axis=0)
            for e in x:
                if e.item():
                    temp.append(e.item() / data.edge_attr.shape[0])
                else:
                    temp.append(e.item())
            feature[i] = np.array(temp)
        return feature

    def get_samples(self, minority_samples):

        feature = self.get_feature(minority_samples)
        # 找出某一类样本集(数据集X)中的每一个样本在数据集X中的k个近邻
        knn = NearestNeighbors(n_neighbors=self.k).fit(feature)

        for i in range(len(minority_samples)):

            k_neighbors = knn.kneighbors(feature[i].reshape(1, -1), return_distance=False)[0]
            # 对某一类样本集(minority class samples)中每个样本, 分别根据其k个近邻生成

            # sampling_rate个新的样本
            for j in range(self.sampling_rate):
                # 从k个近邻里面随机选择一个近邻
                neighbor = np.random.choice(k_neighbors)

                # diff_x
                l1 = minority_samples[neighbor].x.shape[0]
                l2 = minority_samples[i].x.shape[0]
                # 计算样本X[i]与刚刚选择的近邻的差
                if l1 >= l2:
                    diff_x = minority_samples[neighbor].x[:l2, :] - minority_samples[i].x
                else:
                    zero_x = torch.zeros(l2 - l1, minority_samples[i].x.shape[1])
                    diff_x = torch.cat(((minority_samples[neighbor].x - minority_samples[i].x[:l1, :]), zero_x), dim=0)

                # diff_attr
                l1 = minority_samples[neighbor].edge_attr.shape[0]
                l2 = minority_samples[i].edge_attr.shape[0]
                # 计算样本edge_attr[i]与刚刚选择的近邻的差
                if l1 >= l2:
                    diff_attr = minority_samples[neighbor].edge_attr[:l2, :] - minority_samples[i].edge_attr
                else:
                    zero_attr = torch.zeros(l2 - l1, minority_samples[i].edge_attr.shape[1])
                    diff_attr = torch.cat(
                        ((minority_samples[neighbor].edge_attr - minority_samples[i].edge_attr[:l1, :]), zero_attr),
                        dim=0)

                # 生成新的数据
                self.smote_samples.append(Data(x=(minority_samples[i].x + random.random() * diff_x),
                                                edge_index=minority_samples[i].edge_index,
                                                edge_attr=(minority_samples[i].edge_attr + random.random() * diff_attr),
                                                y=minority_samples[i].y))
        feature1 = self.get_feature(self.smote_samples)

        def distij(i, j):
            return np.sqrt(np.sum(np.square(feature1[i] - feature1[j])))

        picker = MaxMinPicker()
        print(len(self.smote_samples), self.more - self.less)
        pickIndices = picker.LazyPick(distij, len(self.smote_samples), self.more - self.less, seed=1)
        for x in pickIndices:
            self.datasets_smote.append(self.smote_samples[x])
        end = time.time()
        # print(end - start, "s")

class SmoteSimilarity:

    """
    SMOTE过采样算法.

    Parameters:
    -----------
    k: int
        选取的近邻数目.
    sampling_rate: int
        采样倍数, attention sampling_rate < k.
    newindex: int
        生成的新样本(合成样本)的索引号.
    """

    def __init__(self, k=5):
        self.sampling_rate = 1
        self.k = k

    def fit(self, datasets,mols):
        self.datasets_smote = datasets  # 浅复制
        self.mols = mols

        positive = list()
        negative = list()
        positive_mols = list()
        negative_mols = list()

        for data,mol1 in zip(self.datasets_smote, mols):
            if data.y == 1:
                positive.append(data)
                positive_mols.append(mol1)
            else:
                negative.append(data)
                negative_mols.append(mol1)

        len_negative = len(negative)
        len_positive = len(positive)

        print("smote_fp:k=%d nontox_idx:%d tox_idx:%d" % (self.k, len_negative, len_positive))

        if abs(len_positive - len_negative) <= (min(len_positive, len_negative) * 0.5):
            return self.datasets_smote
        elif len_positive < len_negative:
            self.sampling_rate = 1 if (((len_negative - len_positive) // len_positive) == 0) else (
                        (len_negative - len_positive) // len_positive)
            print("smote_fp:sampling_rate=%d nontox_idx:%d tox_idx:%d" % (self.sampling_rate, len_negative, len_positive * (self.sampling_rate + 1)))
            self.get_samples(positive, positive_mols)
            return self.datasets_smote
        else:
            self.sampling_rate = 1 if (((len_positive - len_negative) // len_negative) == 0) else (
                        (len_positive - len_negative) // len_negative)
            print("smote_fp:sampling_rate=%d nontox_idx:%d tox_idx:%d" % (self.sampling_rate, len_negative * (self.sampling_rate + 1), len_positive))
            self.get_samples(negative, negative_mols)
            return self.datasets_smote

    def get_samples(self, minority_samples, minority_samples_mols):
        # 最大特征长度
        # feature_size = 0
        # for data in minority_samples:
        #     temp = np.append([], data.x)
        #     if feature_size < temp.size:
        #         feature_size = temp.size
        # print('smote:feature_size_max:', feature_size)
        #
        # feature = np.zeros(shape=(len(minority_samples), feature_size))
        # for i, data in enumerate(minority_samples):
        #     temp = np.append([], data.x)
        #     feature[i] = np.pad(temp, (0, feature_size - temp.size), 'constant')
        #
        # # 找出某一类样本集(数据集X)中的每一个样本在数据集X中的k个近邻
        # knn = NearestNeighbors(n_neighbors=self.k).fit(feature)
        start = time.time()
        fps_MACCS = list()
        fps_Morgan = list()
        for mol in minority_samples_mols:
            fps_MACCS.append(MACCSkeys.GenMACCSKeys(mol))
            fps_Morgan.append(SimilarityMaps.GetMorganFingerprint(mol, fpType='bv'))

        k_neighbors = list()
        for i in range(len(minority_samples_mols)):
            similarity = {}
            for j in range(len(minority_samples_mols)):
                if i != j:
                    s1 = DataStructs.FingerprintSimilarity(fps_MACCS[i], fps_MACCS[j])
                    s2 = DataStructs.FingerprintSimilarity(fps_Morgan[i], fps_Morgan[j])
                    similarity[j] = np.mean([s1, s2])
            new = sorted(similarity.items(), key=lambda d: d[1], reverse=True)
            j = 0
            neighbors = list()
            for data in new:
                neighbors.append(data[0])
                j = j + 1
                if j == 5:
                    break
            k_neighbors.append(neighbors)
        end = time.time()
        print(end - start, "s")
        for i in range(len(minority_samples)):

            # k_neighbors = knn.kneighbors(feature[i].reshape(1, -1), return_distance=False)[0]
            # 对某一类样本集(minority class samples)中每个样本, 分别根据其k个近邻生成

            # sampling_rate个新的样本
            for j in range(self.sampling_rate):
                # 从k个近邻里面随机选择一个近邻
                neighbor = np.random.choice(k_neighbors[i])
                l1 = minority_samples[neighbor].x.shape[0]
                l2 = minority_samples[i].x.shape[0]
                # 计算样本X[i]与刚刚选择的近邻的差
                if l1 >= l2:
                    diff = minority_samples[neighbor].x[:l2, :] - minority_samples[i].x
                else:
                    zero = torch.zeros(l2 - l1, minority_samples[i].x.shape[1])
                    diff = torch.cat(((minority_samples[neighbor].x - minority_samples[i].x[:l1, :]), zero), dim=0)
                # 生成新的数据
                self.datasets_smote.append(Data(x=(minority_samples[i].x + random.random() * diff),
                                                edge_index=minority_samples[i].edge_index,
                                                edge_attr=minority_samples[i].edge_attr,
                                                y=minority_samples[i].y))


class SmoteSimilarityAllFPXAttrFP:
    """
    SMOTE过采样算法.

    Parameters:
    -----------
    k: int
        选取的近邻数目.
    sampling_rate: int
        采样倍数, attention sampling_rate < k.
    newindex: int
        生成的新样本(合成样本)的索引号.
    """

    def __init__(self, k=5):
        self.sampling_rate = 1
        self.k = k

    def fit(self, datasets,mols):
        self.datasets_smote = datasets  # 浅复制
        self.mols = mols

        positive = list()
        negative = list()
        positive_mols = list()
        negative_mols = list()

        for data,mol1 in zip(self.datasets_smote, mols):
            if data.y == 1:
                positive.append(data)
                positive_mols.append(mol1)
            else:
                negative.append(data)
                negative_mols.append(mol1)

        len_negative = len(negative)
        len_positive = len(positive)

        print("smote_fp_all_fp:k=%d nontox_idx:%d tox_idx:%d" % (self.k, len_negative, len_positive))

        if len_positive == len_negative:
            return self.datasets_smote
        elif len_positive < len_negative:
            self.more = len_negative
            self.less = len_positive
            self.sampling_rate = 1 +(len_negative - len_positive) // len_positive
            print("smote_fp_all_fp:sampling_rate=%d nontox_idx:%d tox_idx:%d" % (self.sampling_rate, len_negative, len_positive * (self.sampling_rate + 1)))
            self.get_samples(positive, positive_mols)
            return self.datasets_smote
        else:
            self.more = len_positive
            self.less = len_negative
            self.sampling_rate = 1 + (len_positive - len_negative) // len_negative
            print("smote_fp_all_fp:sampling_rate=%d nontox_idx:%d tox_idx:%d" % (self.sampling_rate, len_negative * (self.sampling_rate + 1), len_positive))
            self.get_samples(negative, negative_mols)
            return self.datasets_smote

    def get_feature(self, samples):
        feature = np.zeros(shape=(len(samples), 52 + 10))
        for i, data in enumerate(samples):
            temp = list()
            x = data.x.sum(axis=0)
            for e in x:
                if e.item():
                    temp.append(e.item() / data.x.shape[0])
                else:
                    temp.append(e.item())
            x = data.edge_attr.sum(axis=0)
            for e in x:
                if e.item():
                    temp.append(e.item() / data.edge_attr.shape[0])
                else:
                    temp.append(e.item())
            feature[i] = np.array(temp)
        return feature

    def get_samples(self, minority_samples, minority_samples_mols):
        # 最大特征长度
        # feature_size = 0
        # for data in minority_samples:
        #     temp = np.append([], data.x)
        #     if feature_size < temp.size:
        #         feature_size = temp.size
        # print('smote:feature_size_max:', feature_size)
        #
        # feature = np.zeros(shape=(len(minority_samples), feature_size))
        # for i, data in enumerate(minority_samples):
        #     temp = np.append([], data.x)
        #     feature[i] = np.pad(temp, (0, feature_size - temp.size), 'constant')
        #
        # # 找出某一类样本集(数据集X)中的每一个样本在数据集X中的k个近邻
        # knn = NearestNeighbors(n_neighbors=self.k).fit(feature)
        start = time.time()
        # 处理自身的特征，算出两个分子间的欧式距离的最大值和最小值
        feature = self.get_feature(minority_samples)
        max_dis = np.sqrt(np.sum(np.square(feature[0] - feature[1])))
        min_dis = np.sqrt(np.sum(np.square(feature[0] - feature[1])))
        for i in range(len(minority_samples_mols)):
            for j in range(len(minority_samples_mols)):
                if i != j:
                    dis = np.sqrt(np.sum(np.square(feature[i] - feature[j])))
                    if max_dis < dis:
                        max_dis = dis
                    if min_dis > dis:
                        min_dis = dis

        print("max_dis:{:.4f};min_dis:{:.4f}".format(max_dis,min_dis))

        # 处理四种分子指纹
        fps_MACCS = list()
        fps_Morgan = list()
        fps_Atom_Pairs = list()
        fps_Topological_Torsions = list()
        for mol in minority_samples_mols:
            fps_MACCS.append(MACCSkeys.GenMACCSKeys(mol))
            fps_Morgan.append(SimilarityMaps.GetMorganFingerprint(mol, fpType='bv'))
            fps_Atom_Pairs.append(SimilarityMaps.GetAPFingerprint(mol, fpType='bv'))
            fps_Topological_Torsions.append(SimilarityMaps.GetTTFingerprint(mol, fpType='bv'))

        k_neighbors = list()
        for i in range(len(minority_samples_mols)):
            similarity = {}
            for j in range(len(minority_samples_mols)):
                if i != j:
                    s1 = DataStructs.FingerprintSimilarity(fps_MACCS[i], fps_MACCS[j])
                    s2 = DataStructs.FingerprintSimilarity(fps_Morgan[i], fps_Morgan[j])
                    s3 = DataStructs.FingerprintSimilarity(fps_Atom_Pairs[i], fps_Atom_Pairs[j])
                    s4 = DataStructs.FingerprintSimilarity(fps_Topological_Torsions[i], fps_Topological_Torsions[j])
                    dis = np.sqrt(np.sum(np.square(feature[i] - feature[j])))
                    s5 = 1.0 - (dis-min_dis) / (max_dis-min_dis)
                    similarity[j] = np.mean([s1, s2, s3, s4, s5])
            new = sorted(similarity.items(), key=lambda d: d[1], reverse=True)
            j = 0
            neighbors = list()
            for data in new:
                neighbors.append(data[0])
                j = j + 1
                if j == 5:
                    break
            k_neighbors.append(neighbors)


        end = time.time()
        print(end - start, "s")
        for i in range(len(minority_samples)):

            # k_neighbors = knn.kneighbors(feature[i].reshape(1, -1), return_distance=False)[0]
            # 对某一类样本集(minority class samples)中每个样本, 分别根据其k个近邻生成

            # sampling_rate个新的样本
            for j in range(self.sampling_rate):
                # 从k个近邻里面随机选择一个近邻
                neighbor = np.random.choice(k_neighbors[i])

                # diff_x
                l1 = minority_samples[neighbor].x.shape[0]
                l2 = minority_samples[i].x.shape[0]
                # 计算样本X[i]与刚刚选择的近邻的差
                if l1 >= l2:
                    diff_x = minority_samples[neighbor].x[:l2, :] - minority_samples[i].x
                else:
                    zero_x = torch.zeros(l2 - l1, minority_samples[i].x.shape[1])
                    diff_x = torch.cat(((minority_samples[neighbor].x - minority_samples[i].x[:l1, :]), zero_x), dim=0)

                # diff_attr
                l1 = minority_samples[neighbor].edge_attr.shape[0]
                l2 = minority_samples[i].edge_attr.shape[0]
                # 计算样本edge_attr[i]与刚刚选择的近邻的差
                if l1 >= l2:
                    diff_attr = minority_samples[neighbor].edge_attr[:l2, :] - minority_samples[i].edge_attr
                else:
                    zero_attr = torch.zeros(l2 - l1, minority_samples[i].edge_attr.shape[1])
                    diff_attr = torch.cat(
                        ((minority_samples[neighbor].edge_attr - minority_samples[i].edge_attr[:l1, :]), zero_attr),
                        dim=0)

                # 生成新的数据
                self.datasets_smote.append(Data(x=(minority_samples[i].x + random.random() * diff_x),
                                                edge_index=minority_samples[i].edge_index,
                                                edge_attr=(minority_samples[i].edge_attr + random.random() * diff_attr),
                                                y=minority_samples[i].y))

class SmoteSimilarityAllFPXAttrMaxMiin:
    """
    SMOTE过采样算法.

    Parameters:
    -----------
    k: int
        选取的近邻数目.
    sampling_rate: int
        采样倍数, attention sampling_rate < k.
    newindex: int
        生成的新样本(合成样本)的索引号.
    """

    def __init__(self, k=5):
        self.sampling_rate = 1
        self.k = k

    def fit(self, datasets,mols):
        self.datasets_smote = datasets  # 浅复制
        self.mols = mols

        self.smote_samples = list()

        positive = list()
        negative = list()
        positive_mols = list()
        negative_mols = list()

        for data,mol1 in zip(self.datasets_smote, mols):
            if data.y == 1:
                positive.append(data)
                positive_mols.append(mol1)
            else:
                negative.append(data)
                negative_mols.append(mol1)

        len_negative = len(negative)
        len_positive = len(positive)

        print("smote_fp_all_MaxMin:k=%d nontox_idx:%d tox_idx:%d" % (self.k, len_negative, len_positive))

        if len_positive == len_negative:
            return self.datasets_smote
        elif len_positive < len_negative:
            self.more = len_negative
            self.less = len_positive
            self.sampling_rate = 1 +(len_negative - len_positive) // len_positive
            print("smote_fp_all_MaxMin:sampling_rate=%d nontox_idx:%d tox_idx:%d" % (self.sampling_rate, len_negative, len_positive * (self.sampling_rate + 1)))
            self.get_samples(positive, positive_mols)
            return self.datasets_smote
        else:
            self.more = len_positive
            self.less = len_negative
            self.sampling_rate = 1 + (len_positive - len_negative) // len_negative
            print("smote_fp_all_MaxMin:sampling_rate=%d nontox_idx:%d tox_idx:%d" % (self.sampling_rate, len_negative * (self.sampling_rate + 1), len_positive))
            self.get_samples(negative, negative_mols)
            return self.datasets_smote

    def get_feature(self, samples):
        feature = np.zeros(shape=(len(samples), 52 + 10))
        for i, data in enumerate(samples):
            temp = list()
            x = data.x.sum(axis=0)
            for e in x:
                if e.item():
                    temp.append(e.item() / data.x.shape[0])
                else:
                    temp.append(e.item())
            x = data.edge_attr.sum(axis=0)
            for e in x:
                if e.item():
                    temp.append(e.item() / data.edge_attr.shape[0])
                else:
                    temp.append(e.item())
            feature[i] = np.array(temp)
        return feature

    def get_samples(self, minority_samples, minority_samples_mols):
        # 最大特征长度
        # feature_size = 0
        # for data in minority_samples:
        #     temp = np.append([], data.x)
        #     if feature_size < temp.size:
        #         feature_size = temp.size
        # print('smote:feature_size_max:', feature_size)
        #
        # feature = np.zeros(shape=(len(minority_samples), feature_size))
        # for i, data in enumerate(minority_samples):
        #     temp = np.append([], data.x)
        #     feature[i] = np.pad(temp, (0, feature_size - temp.size), 'constant')
        #
        # # 找出某一类样本集(数据集X)中的每一个样本在数据集X中的k个近邻
        # knn = NearestNeighbors(n_neighbors=self.k).fit(feature)
        start = time.time()
        # 处理自身的特征，算出两个分子间的欧式距离的最大值和最小值
        feature = self.get_feature(minority_samples)
        max_dis = np.sqrt(np.sum(np.square(feature[0] - feature[1])))
        min_dis = np.sqrt(np.sum(np.square(feature[0] - feature[1])))
        for i in range(len(minority_samples_mols)):
            for j in range(len(minority_samples_mols)):
                if i != j:
                    dis = np.sqrt(np.sum(np.square(feature[i] - feature[j])))
                    if max_dis < dis:
                        max_dis = dis
                    if min_dis > dis:
                        min_dis = dis

        print("minority_samples : max_dis:{:.4f};min_dis:{:.4f}".format(max_dis,min_dis))

        # 处理四种分子指纹
        fps_MACCS = list()
        fps_Morgan = list()
        fps_Atom_Pairs = list()
        fps_Topological_Torsions = list()
        for mol in minority_samples_mols:
            fps_MACCS.append(MACCSkeys.GenMACCSKeys(mol))
            fps_Morgan.append(SimilarityMaps.GetMorganFingerprint(mol, fpType='bv'))
            fps_Atom_Pairs.append(SimilarityMaps.GetAPFingerprint(mol, fpType='bv'))
            fps_Topological_Torsions.append(SimilarityMaps.GetTTFingerprint(mol, fpType='bv'))

        k_neighbors = list()
        for i in range(len(minority_samples_mols)):
            similarity = {}
            for j in range(len(minority_samples_mols)):
                if i != j:
                    s1 = DataStructs.FingerprintSimilarity(fps_MACCS[i], fps_MACCS[j])
                    s2 = DataStructs.FingerprintSimilarity(fps_Morgan[i], fps_Morgan[j])
                    s3 = DataStructs.FingerprintSimilarity(fps_Atom_Pairs[i], fps_Atom_Pairs[j])
                    s4 = DataStructs.FingerprintSimilarity(fps_Topological_Torsions[i], fps_Topological_Torsions[j])
                    dis = np.sqrt(np.sum(np.square(feature[i] - feature[j])))
                    s5 = 1.0 - (dis-min_dis) / (max_dis-min_dis)
                    similarity[j] = np.mean([s1, s2, s3, s4, s5])
            new = sorted(similarity.items(), key=lambda d: d[1], reverse=True)
            j = 0
            neighbors = list()
            for data in new:
                neighbors.append(data[0])
                j = j + 1
                if j == 5:
                    break
            k_neighbors.append(neighbors)

        for i in range(len(minority_samples)):

            # k_neighbors = knn.kneighbors(feature[i].reshape(1, -1), return_distance=False)[0]
            # 对某一类样本集(minority class samples)中每个样本, 分别根据其k个近邻生成

            # sampling_rate个新的样本
            for j in range(self.sampling_rate):
                # 从k个近邻里面随机选择一个近邻
                neighbor = np.random.choice(k_neighbors[i])

                # diff_x
                l1 = minority_samples[neighbor].x.shape[0]
                l2 = minority_samples[i].x.shape[0]
                # 计算样本X[i]与刚刚选择的近邻的差
                if l1 >= l2:
                    diff_x = minority_samples[neighbor].x[:l2, :] - minority_samples[i].x
                else:
                    zero_x = torch.zeros(l2 - l1, minority_samples[i].x.shape[1])
                    diff_x = torch.cat(((minority_samples[neighbor].x - minority_samples[i].x[:l1, :]), zero_x), dim=0)

                # diff_attr
                l1 = minority_samples[neighbor].edge_attr.shape[0]
                l2 = minority_samples[i].edge_attr.shape[0]
                # 计算样本edge_attr[i]与刚刚选择的近邻的差
                if l1 >= l2:
                    diff_attr = minority_samples[neighbor].edge_attr[:l2, :] - minority_samples[i].edge_attr
                else:
                    zero_attr = torch.zeros(l2 - l1, minority_samples[i].edge_attr.shape[1])
                    diff_attr = torch.cat(
                        ((minority_samples[neighbor].edge_attr - minority_samples[i].edge_attr[:l1, :]), zero_attr),
                        dim=0)

                # 生成新的数据
                self.smote_samples.append(Data(x=(minority_samples[i].x + random.random() * diff_x),
                                                edge_index=minority_samples[i].edge_index,
                                                edge_attr=(minority_samples[i].edge_attr + random.random() * diff_attr),
                                                y=minority_samples[i].y))

        feature1 = self.get_feature(self.smote_samples)
        # max_dis = np.sqrt(np.sum(np.square(feature[0] - feature[1])))
        # min_dis = np.sqrt(np.sum(np.square(feature[0] - feature[1])))
        # for i in range(len(minority_samples_mols)):
        #     for j in range(len(minority_samples_mols)):
        #         if i != j:
        #             dis = np.sqrt(np.sum(np.square(feature[i] - feature[j])))
        #             if max_dis < dis:
        #                 max_dis = dis
        #             if min_dis > dis:
        #                 min_dis = dis
        #
        # print("smote_samples : max_dis:{:.4f};min_dis:{:.4f}".format(max_dis,min_dis))
        #
        # # 处理四种分子指纹
        # fps_MACCS = list()
        # fps_Morgan = list()
        # fps_Atom_Pairs = list()
        # fps_Topological_Torsions = list()
        # for mol in minority_samples_mols:
        #     fps_MACCS.append(MACCSkeys.GenMACCSKeys(mol))
        #     fps_Morgan.append(SimilarityMaps.GetMorganFingerprint(mol, fpType='bv'))
        #     fps_Atom_Pairs.append(SimilarityMaps.GetAPFingerprint(mol, fpType='bv'))
        #     fps_Topological_Torsions.append(SimilarityMaps.GetTTFingerprint(mol, fpType='bv'))

        def distij(i, j):
            # s1 = DataStructs.FingerprintSimilarity(fps_MACCS[i], fps_MACCS[j])
            # s2 = DataStructs.FingerprintSimilarity(fps_Morgan[i], fps_Morgan[j])
            # s3 = DataStructs.FingerprintSimilarity(fps_Atom_Pairs[i], fps_Atom_Pairs[j])
            # s4 = DataStructs.FingerprintSimilarity(fps_Topological_Torsions[i], fps_Topological_Torsions[j])
            # dis = np.sqrt(np.sum(np.square(feature1[i] - feature1[j])))
            # s5 = 1.0 - (dis - min_dis) / (max_dis - min_dis)
            return np.sqrt(np.sum(np.square(feature1[i] - feature1[j])))

        picker = MaxMinPicker()
        pickIndices = picker.LazyPick(distij, len(self.smote_samples), self.more - self.less, seed=1)
        for x in pickIndices:
            self.datasets_smote.append(self.smote_samples[x])
        end = time.time()
        print(end - start, "s")


class SmoteSimilarityAllFPXAttrAvgGCN:
    """
    SMOTE过采样算法.

    Parameters:
    -----------
    k: int
        选取的近邻数目.
    sampling_rate: int
        采样倍数, attention sampling_rate < k.
    newindex: int
        生成的新样本(合成样本)的索引号.
    """

    def __init__(self, k=5):
        self.sampling_rate = 1
        self.k = k

    def fit(self, datasets,mols):
        self.datasets_smote = datasets  # 浅复制
        self.mols = mols

        self.smote_samples = list()

        positive = list()
        negative = list()
        positive_mols = list()
        negative_mols = list()

        for data,mol1 in zip(self.datasets_smote, mols):
            if data.y == 1:
                positive.append(data)
                positive_mols.append(mol1)
            else:
                negative.append(data)
                negative_mols.append(mol1)

        len_negative = len(negative)
        len_positive = len(positive)

        print("smote_fp_all_x_attr_avg_gcn:k=%d nontox_idx:%d tox_idx:%d" % (self.k, len_negative, len_positive))

        if len_positive == len_negative:
            return self.datasets_smote
        elif len_positive < len_negative:
            self.sampling_rate = 1 + (len_negative - len_positive) // len_positive
            print("smote_fp_all_x_attr_avg_gcn:sampling_rate=%d nontox_idx:%d tox_idx:%d" % (self.sampling_rate, len_negative, len_positive * (self.sampling_rate + 1)))
            self.get_samples(positive, positive_mols)
            return self.datasets_smote
        else:
            self.sampling_rate = 1 + (len_positive - len_negative) // len_negative
            print("smote_fp_all_x_attr_avg_gcn:sampling_rate=%d nontox_idx:%d tox_idx:%d" % (self.sampling_rate, len_negative * (self.sampling_rate + 1), len_positive))
            self.get_samples(negative, negative_mols)
            return self.datasets_smote

    def get_feature(self, samples):
        feature = np.zeros(shape=(len(samples), 52 + 10))
        for i, data in enumerate(samples):
            temp = list()
            x = data.x.sum(axis=0)
            for e in x:
                if e.item():
                    temp.append(e.item() / data.x.shape[0])
                else:
                    temp.append(e.item())
            x = data.edge_attr.sum(axis=0)
            for e in x:
                if e.item():
                    temp.append(e.item() / data.edge_attr.shape[0])
                else:
                    temp.append(e.item())
            feature[i] = np.array(temp)
        return feature

    def get_samples(self, minority_samples, minority_samples_mols):
        # 最大特征长度
        # feature_size = 0
        # for data in minority_samples:
        #     temp = np.append([], data.x)
        #     if feature_size < temp.size:
        #         feature_size = temp.size
        # print('smote:feature_size_max:', feature_size)
        #
        # feature = np.zeros(shape=(len(minority_samples), feature_size))
        # for i, data in enumerate(minority_samples):
        #     temp = np.append([], data.x)
        #     feature[i] = np.pad(temp, (0, feature_size - temp.size), 'constant')
        #
        # # 找出某一类样本集(数据集X)中的每一个样本在数据集X中的k个近邻
        # knn = NearestNeighbors(n_neighbors=self.k).fit(feature)
        start = time.time()
        # 处理自身的特征，算出两个分子间的欧式距离的最大值和最小值
        feature = self.get_feature(minority_samples)
        max_dis = np.sqrt(np.sum(np.square(feature[0] - feature[1])))
        min_dis = np.sqrt(np.sum(np.square(feature[0] - feature[1])))
        for i in range(len(minority_samples_mols)):
            for j in range(len(minority_samples_mols)):
                if i != j:
                    dis = np.sqrt(np.sum(np.square(feature[i] - feature[j])))
                    if max_dis < dis:
                        max_dis = dis
                    if min_dis > dis:
                        min_dis = dis

        print("max_dis:{:.4f};min_dis:{:.4f}".format(max_dis,min_dis))

        # 处理四种分子指纹
        fps_MACCS = list()
        fps_Morgan = list()
        fps_Atom_Pairs = list()
        fps_Topological_Torsions = list()
        for mol in minority_samples_mols:
            fps_MACCS.append(MACCSkeys.GenMACCSKeys(mol))
            fps_Morgan.append(SimilarityMaps.GetMorganFingerprint(mol, fpType='bv'))
            fps_Atom_Pairs.append(SimilarityMaps.GetAPFingerprint(mol, fpType='bv'))
            fps_Topological_Torsions.append(SimilarityMaps.GetTTFingerprint(mol, fpType='bv'))

        k_neighbors = list()
        for i in range(len(minority_samples_mols)):
            similarity = {}
            for j in range(len(minority_samples_mols)):
                if i != j:
                    s1 = DataStructs.FingerprintSimilarity(fps_MACCS[i], fps_MACCS[j])
                    s2 = DataStructs.FingerprintSimilarity(fps_Morgan[i], fps_Morgan[j])
                    s3 = DataStructs.FingerprintSimilarity(fps_Atom_Pairs[i], fps_Atom_Pairs[j])
                    s4 = DataStructs.FingerprintSimilarity(fps_Topological_Torsions[i], fps_Topological_Torsions[j])
                    dis = np.sqrt(np.sum(np.square(feature[i] - feature[j])))
                    s5 = 1.0 - (dis-min_dis) / (max_dis-min_dis)
                    similarity[j] = np.mean([s1, s2, s3, s4, s5])
            new = sorted(similarity.items(), key=lambda d: d[1], reverse=True)
            j = 0
            neighbors = list()
            for data in new:
                neighbors.append(data[0])
                j = j + 1
                if j == 5:
                    break
            k_neighbors.append(neighbors)

        end = time.time()
        print(end - start, "s")

        for i in range(len(minority_samples)):

            # k_neighbors = knn.kneighbors(feature[i].reshape(1, -1), return_distance=False)[0]
            # 对某一类样本集(minority class samples)中每个样本, 分别根据其k个近邻生成

            # sampling_rate个新的样本
            for j in range(self.sampling_rate):
                # 从k个近邻里面随机选择一个近邻
                neighbor = np.random.choice(k_neighbors[i])

                # diff_x
                temp = list()
                x = minority_samples[neighbor].x.sum(axis=0)
                for e in x:
                    if e.item():
                        temp.append(e.item() / minority_samples[neighbor].x.shape[0])
                    else:
                        temp.append(e.item())
                x_avg = torch.Tensor([temp for k in range(minority_samples[i].x.shape[0])])
                # print("x_avg:",x_avg.shape,minority_samples[i].x.shape)
                # 计算样本X[i]与刚刚选择的近邻的差
                diff_x = x_avg - minority_samples[i].x

                # diff_attr
                temp = list()
                edge_attr = minority_samples[neighbor].edge_attr.sum(axis=0)
                for e in edge_attr:
                    if e.item():
                        temp.append(e.item() / minority_samples[neighbor].edge_attr.shape[0])
                    else:
                        temp.append(e.item())
                if minority_samples[i].edge_attr.shape[0] != 0:
                    attr_avg = torch.Tensor([temp for k in range(minority_samples[i].edge_attr.shape[0])])
                else:
                    attr_avg = torch.zeros(0, 10, dtype=torch.float)
                # print("attr_avg:",attr_avg.shape,minority_samples[i].edge_attr.shape)
                # 计算样本X[i]与刚刚选择的近邻的差
                diff_attr = attr_avg - minority_samples[i].edge_attr

                # 生成新的数据
                self.datasets_smote.append(Data(x=(minority_samples[i].x + random.random() * diff_x),
                                                edge_index=minority_samples[i].edge_index,
                                                edge_attr=(minority_samples[i].edge_attr + random.random() * diff_attr),
                                                y=minority_samples[i].y))

class SmoteSimilarityAllFPXAttrAvgMaxMin:
    """
    SMOTE过采样算法.

    Parameters:
    -----------
    k: int
        选取的近邻数目.
    sampling_rate: int
        采样倍数, attention sampling_rate < k.
    newindex: int
        生成的新样本(合成样本)的索引号.
    """

    def __init__(self, k=5):
        self.sampling_rate = 1
        self.k = k

    def fit(self, datasets,mols):
        self.datasets_smote = datasets  # 浅复制
        self.mols = mols

        self.smote_samples = list()

        positive = list()
        negative = list()
        positive_mols = list()
        negative_mols = list()

        for data,mol1 in zip(self.datasets_smote, mols):
            if data.y == 1:
                positive.append(data)
                positive_mols.append(mol1)
            else:
                negative.append(data)
                negative_mols.append(mol1)

        len_negative = len(negative)
        len_positive = len(positive)

        print("smote_fp_all_x_attr_avg_MaxMin:k=%d nontox_idx:%d tox_idx:%d" % (self.k, len_negative, len_positive))

        if len_positive == len_negative:
            return self.datasets_smote
        elif len_positive < len_negative:
            self.more = len_negative
            self.less = len_positive
            self.sampling_rate = 1 +(len_negative - len_positive) // len_positive
            print("smote_fp_all_x_attr_avg_MaxMin:sampling_rate=%d nontox_idx:%d tox_idx:%d" % (self.sampling_rate, len_negative, len_positive * (self.sampling_rate + 1)))
            self.get_samples(positive, positive_mols)
            return self.datasets_smote
        else:
            self.more = len_positive
            self.less = len_negative
            self.sampling_rate = 1 + (len_positive - len_negative) // len_negative
            print("smote_fp_all_x_attr_avg_MaxMin:sampling_rate=%d nontox_idx:%d tox_idx:%d" % (self.sampling_rate, len_negative * (self.sampling_rate + 1), len_positive))
            self.get_samples(negative, negative_mols)
            return self.datasets_smote

    def get_feature(self, samples):
        feature = np.zeros(shape=(len(samples), 52 + 10))
        for i, data in enumerate(samples):
            temp = list()
            x = data.x.sum(axis=0)
            for e in x:
                if e.item():
                    temp.append(e.item() / data.x.shape[0])
                else:
                    temp.append(e.item())
            x = data.edge_attr.sum(axis=0)
            for e in x:
                if e.item():
                    temp.append(e.item() / data.edge_attr.shape[0])
                else:
                    temp.append(e.item())
            feature[i] = np.array(temp)
        return feature

    def get_samples(self, minority_samples, minority_samples_mols):
        # 最大特征长度
        # feature_size = 0
        # for data in minority_samples:
        #     temp = np.append([], data.x)
        #     if feature_size < temp.size:
        #         feature_size = temp.size
        # print('smote:feature_size_max:', feature_size)
        #
        # feature = np.zeros(shape=(len(minority_samples), feature_size))
        # for i, data in enumerate(minority_samples):
        #     temp = np.append([], data.x)
        #     feature[i] = np.pad(temp, (0, feature_size - temp.size), 'constant')
        #
        # # 找出某一类样本集(数据集X)中的每一个样本在数据集X中的k个近邻
        # knn = NearestNeighbors(n_neighbors=self.k).fit(feature)
        start = time.time()
        # 处理自身的特征，算出两个分子间的欧式距离的最大值和最小值
        feature = self.get_feature(minority_samples)
        max_dis = np.sqrt(np.sum(np.square(feature[0] - feature[1])))
        min_dis = np.sqrt(np.sum(np.square(feature[0] - feature[1])))
        for i in range(len(minority_samples_mols)):
            for j in range(len(minority_samples_mols)):
                if i != j:
                    dis = np.sqrt(np.sum(np.square(feature[i] - feature[j])))
                    if max_dis < dis:
                        max_dis = dis
                    if min_dis > dis:
                        min_dis = dis

        print("max_dis:{:.4f};min_dis:{:.4f}".format(max_dis,min_dis))

        # 处理四种分子指纹
        fps_MACCS = list()
        fps_Morgan = list()
        fps_Atom_Pairs = list()
        fps_Topological_Torsions = list()
        for mol in minority_samples_mols:
            fps_MACCS.append(MACCSkeys.GenMACCSKeys(mol))
            fps_Morgan.append(SimilarityMaps.GetMorganFingerprint(mol, fpType='bv'))
            fps_Atom_Pairs.append(SimilarityMaps.GetAPFingerprint(mol, fpType='bv'))
            fps_Topological_Torsions.append(SimilarityMaps.GetTTFingerprint(mol, fpType='bv'))

        k_neighbors = list()
        for i in range(len(minority_samples_mols)):
            similarity = {}
            for j in range(len(minority_samples_mols)):
                if i != j:
                    s1 = DataStructs.FingerprintSimilarity(fps_MACCS[i], fps_MACCS[j])
                    s2 = DataStructs.FingerprintSimilarity(fps_Morgan[i], fps_Morgan[j])
                    s3 = DataStructs.FingerprintSimilarity(fps_Atom_Pairs[i], fps_Atom_Pairs[j])
                    s4 = DataStructs.FingerprintSimilarity(fps_Topological_Torsions[i], fps_Topological_Torsions[j])
                    dis = np.sqrt(np.sum(np.square(feature[i] - feature[j])))
                    s5 = 1.0 - (dis-min_dis) / (max_dis-min_dis)
                    similarity[j] = np.mean([s1, s2, s3, s4, s5])
            new = sorted(similarity.items(), key=lambda d: d[1], reverse=True)
            j = 0
            neighbors = list()
            for data in new:
                neighbors.append(data[0])
                j = j + 1
                if j == 5:
                    break
            k_neighbors.append(neighbors)

        for i in range(len(minority_samples)):

            # k_neighbors = knn.kneighbors(feature[i].reshape(1, -1), return_distance=False)[0]
            # 对某一类样本集(minority class samples)中每个样本, 分别根据其k个近邻生成

            # sampling_rate个新的样本
            for j in range(self.sampling_rate):
                # 从k个近邻里面随机选择一个近邻
                neighbor = np.random.choice(k_neighbors[i])

                # diff_x
                temp = list()
                x = minority_samples[neighbor].x.sum(axis=0)
                for e in x:
                    if e.item():
                        temp.append(e.item() / minority_samples[neighbor].x.shape[0])
                    else:
                        temp.append(e.item())
                x_avg = torch.Tensor([temp for k in range(minority_samples[i].x.shape[0])])
                # print(x_avg.shape,minority_samples[i].x.shape)
                # 计算样本X[i]与刚刚选择的近邻的差
                diff_x = x_avg - minority_samples[i].x

                # diff_attr
                temp = list()
                edge_attr = minority_samples[neighbor].edge_attr.sum(axis=0)
                for e in edge_attr:
                    if e.item():
                        temp.append(e.item() / minority_samples[neighbor].edge_attr.shape[0])
                    else:
                        temp.append(e.item())
                if minority_samples[i].edge_attr.shape[0] != 0:
                    attr_avg = torch.Tensor([temp for k in range(minority_samples[i].edge_attr.shape[0])])
                else:
                    attr_avg = torch.zeros(0, 10, dtype=torch.float)
                # 计算样本edge_attr[i]与刚刚选择的近邻的差
                diff_attr = attr_avg - minority_samples[i].edge_attr

                # 生成新的数据
                self.smote_samples.append(Data(x=(minority_samples[i].x + random.random() * diff_x),
                                                edge_index=minority_samples[i].edge_index,
                                                edge_attr=(minority_samples[i].edge_attr + random.random() * diff_attr),
                                                y=minority_samples[i].y))
        feature1 = self.get_feature(self.smote_samples)

        def distij(i, j):
            return np.sqrt(np.sum(np.square(feature1[i] - feature1[j])))

        picker = MaxMinPicker()
        pickIndices = picker.LazyPick(distij, len(self.smote_samples), self.more - self.less, seed=1)
        for x in pickIndices:
            self.datasets_smote.append(self.smote_samples[x])
        end = time.time()
        print(end - start, "s")

def random_over_sampling(datasets):
    positive = list()
    negative = list()

    for data in datasets:
        if data.y == 1:
            positive.append(data)
        else:
            negative.append(data)

    len_negative = len(negative)
    len_positive = len(positive)
    print("random_over_sampling:size=%d nontox_idx:%d tox_idx:%d" % (len(datasets), len_negative, len_positive))

    disparity = abs(len_negative - len_positive)
    extend_scale = 1
    extend = int(extend_scale * disparity)
    if len_positive <= len_negative:
        for i in range(extend):
            datasets.append(positive[np.random.choice(len_positive)])
        print("random_over_sampling:size=%d nontox_idx:%d tox_idx:%d" % (
        len(datasets), len_negative, len_positive + extend))
    else:
        for i in range(extend):
            datasets.append(negative[np.random.choice(len_negative)])
        print("random_over_sampling:size=%d nontox_idx:%d tox_idx:%d" % (
        len(datasets), len_negative + extend, len_positive))
    return datasets


if '__main__' == __name__:
    datasets = pd.read_csv('./data/raw/nr-ahr.smiles', sep='\t', header=None, names=["Smiles", "Sample ID", "Label"],
                           encoding='utf-8').reset_index()
    num = 0
    fps = list()
    mols = list()
    for smile in datasets['Smiles']:
        mol = Chem.MolFromSmiles(smile)
        if mol:
            num += 1
            # fps.append(MACCSkeys.GenMACCSKeys(mol))
            mols.append(mol)
    datasets = [torch.load(f'./data/processed/nr-ahr_data{i}.pt') for i in range(num)]

    print(id(datasets))
    smote = SmoteSimilarityAllFPXAttrAvg(k=5)  # Expected n_neighbors <= n_samples
    datasets = smote.fit(datasets, mols)
    print(len(datasets))
    print(id(datasets))

    # print(id(datasets))
    # datasets = random_over_sampling(datasets)
    # print(id(datasets))
