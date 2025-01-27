import math
import pickle
import torch
import shutil
import numpy as np
from pathlib import Path
from prettytable import PrettyTable
from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import networkx as nx
from networkx.convert_matrix import from_scipy_sparse_matrix
from sklearn.metrics.pairwise import pairwise_distances
from datetime import datetime
import copy
from torch.nn.utils.rnn import pad_sequence

class EarlyStopping:
    def __init__(self, logging,patience=10, delta=0):
        """
        初始化早停机制
        :param patience: 允许多少个epoch没有提升，默认值为10
        :param delta: 最小的改善幅度，只有损失减少超过这个值才认为是改进，默认值为0
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0  # 用于记录没有改善的epoch数
        self.best_score = None
        self.early_stop = False
        self.best_loss = np.inf
        self.logging = logging

    def __call__(self, val_loss):
        """
        每个epoch训练后调用该方法，检查是否触发早停
        :param val_loss: 验证集的损失值
        :return: 如果需要早停，返回True，否则返回False
        """
        if self.best_score is None:
            self.best_score = val_loss
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0  # reset counter if loss improves
        else:
            self.counter += 1
            self.logging.info(f"Early Stopping {self.counter}/{self.patience}")  # 输出没有改善的计数

        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop

def construct_trajpoi_graph(dataset, args=None):
    train_trajs, val_trajs, test_trajs = dataset.train_trajs, dataset.val_trajs, dataset.test_trajs
    whole_trajs = train_trajs + val_trajs + test_trajs

    # W---M trajs*M trajs  H---N pois*M trajs
    x, y = [], []
    x_w,y_w = [],[]
    vocabs = dataset.vocabs
    for idx, traj in enumerate(whole_trajs):
        traj1 = list(set(traj))
        for POI in traj1:
            x.append(idx)
            y.append(vocabs[POI])
        for loc in traj:
            x_w.append(idx)
            y_w.append(vocabs[loc])

    # N_trajs,N_POIS
    # 创建 PyTorch 稀疏矩阵 (COO格式)
    # x和y代表了稀疏矩阵的行列索引，np.ones代表了值
    H_indices = torch.LongTensor([y,x])  # 行列索引
    # 创建稀疏张量
    values = torch.FloatTensor(np.ones(len(x)))  # 稀疏矩阵的非零值
    H_matrix = torch.sparse_coo_tensor(torch.LongTensor([x,y]), values, size=(len(whole_trajs),len(vocabs)))


    # TV_A = csr_matrix((np.ones(len(x_w), dtype=np.float64), (x_w, y_w)))
    # # 计算 TT_A 矩阵
    # TT_A = TV_A.dot(TV_A.T)
    # TT_A = TT_A.tocoo()  # 转换为 COO 格式
    # # 获取行索引、列索引和数据
    # indices_w = torch.tensor([TT_A.row, TT_A.col], dtype=torch.long)
    # values_w = torch.tensor(TT_A.data, dtype=torch.float32)
    # shape_w = TT_A.shape
    # #构建轨迹-轨迹矩阵
    # W_matrix = torch.sparse_coo_tensor(indices_w, values_w, torch.Size(shape_w))

    return {"H":H_indices,"W":None,"H_matrix":H_matrix}

def construct_paddedtrajs(init_dataset):
    trajs = []
    trajs_len = []
    trajs_hour = []
    trajs_week = []
    geovocab = init_dataset.vidx_to_geoid
    for traj in init_dataset.train_trajs+init_dataset.val_trajs+init_dataset.test_trajs:
        trajs.append(torch.tensor([geovocab[loc] for loc in traj]))
        trajs_len.append(len(traj))
    padded_trajs = pad_sequence(trajs, batch_first=True)
    for traj_hour in init_dataset.train_hours_list+init_dataset.val_hours_list+init_dataset.test_hours_list:
        trajs_hour.append(torch.tensor(traj_hour))
    padded_trajs_hour = pad_sequence(trajs_hour, batch_first=True)
    for traj_week in init_dataset.train_weekdays_list+init_dataset.val_weekdays_list+init_dataset.test_weekdays_list:
        trajs_week.append(torch.tensor(traj_week))
    padded_trajs_week = pad_sequence(trajs_week, batch_first=True)

    return {'trajs':padded_trajs,'trajs_hour':padded_trajs_hour,'trajs_week':padded_trajs_week,'trajs_len':trajs_len}


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        # Computes Macro@F
        pred = output.argmax(-1).squeeze().cpu().numpy()
        macrof = f1_score(target.cpu().numpy(), pred, average="macro")
        res.append(macrof*100)
        macrop = precision_score(target.cpu().numpy(), pred, average="macro")
        res.append(macrop*100)
        macror = recall_score(target.cpu().numpy(), pred, average="macro")
        res.append(macror*100)
        return res