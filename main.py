# coding=utf-8

import torch
import argparse
import time
import os
import logging
import yaml
import datetime
import torch.optim as optim
import random
import numpy as np
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import copy
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import InitDataset,BatchedDataset
from model import HGTUL
# from metrics import batch_performance
from utils import construct_trajpoi_graph,accuracy,EarlyStopping,construct_paddedtrajs
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# 忽略 UndefinedMetricWarning 警告
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# clear cache
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True




# parse argument
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="JKT", help='NYC/TKY/Gowalla')
parser.add_argument('--user_number',type=int,default=500)
parser.add_argument('--seed', default=2024,type=int, help='Random seed')
parser.add_argument('--distance_threshold', default=2.5, type=float, help='distance threshold 2.5 or 0.25')
parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--emb_dim', type=int, default=128, help='embedding size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--decay', type=float, default=5e-4)    # 5e-4
parser.add_argument('--dropout', type=float, default=0.3, help='dropout')    # 0.3
parser.add_argument('--deviceID', type=int, default=0)
# parser.add_argument('--lambda_cl', type=float, default=0.1, help='lambda of contrastive loss')
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--keep_rate', type=float, default=1, help='ratio of edges to keep')
parser.add_argument('--keep_rate_poi', type=float, default=1, help='ratio of poi-poi directed edges to keep')  # 0.7
parser.add_argument('--lr-scheduler-factor', type=float, default=0.1, help='Learning rate scheduler factor')
parser.add_argument('--save_dir', type=str, default="logs")
parser.add_argument('--patience', type=int, default=10)
args = parser.parse_args()

for repeat_idx, seed in enumerate(random.sample(range(0, 1000), 10)):
    args.seed = seed
    print(args.seed)
    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # set device gpu/cpu
    device = torch.device("cuda:{}".format(args.deviceID) if torch.cuda.is_available() else "cpu")

    # set save_dir
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    current_save_dir = os.path.join(args.save_dir, current_time)

    # create current save_dir
    os.mkdir(current_save_dir)

    # Setup logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(current_save_dir, f"log_training.txt"),
                        filemode='w+')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True

    # Save run settings
    args_filename = args.dataset + '_args.yaml'
    with open(os.path.join(current_save_dir, args_filename), 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)
    # Parse Arguments
    logging.info("1. Parse Arguments")
    logging.info(args)
    logging.info("device: {}".format(device))

    # Load Dataset
    logging.info("2. Load Dataset")
    init_dataset = InitDataset(args) 
    args.n_all_trajs = init_dataset.n_all_trajs
    args.n_all_geoids = len(init_dataset.geovocabs)
    NUM_USERS = args.user_number
    NUM_POIS = len(init_dataset.vocabs)
    NUM_TRAJS = args.n_all_trajs
    PADDING_IDX = NUM_POIS
    train_dataset = BatchedDataset(init_dataset.m_train_idx, [init_dataset.train_uid_list[i] for i in init_dataset.m_train_idx])

    logging.info("3. Construct Graph")
    TrajPOIGraph = construct_trajpoi_graph(init_dataset)            # N_trajs, N_POIS ; N_POIS,N_TRAJS
    # construct padded sequence
    padded_data = construct_paddedtrajs(init_dataset)
    

    # 3. Construct DataLoader
    logging.info("4. Construct DataLoader")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Load Model
    logging.info("5. Load Model")
    model = HGTUL(NUM_TRAJS, NUM_POIS, args, device)
    model = model.to(device)
    logging.info(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)

    # Train
    logging.info("6. Start Training")

    best_model = None
    best_acc = -1
    early_stopping = EarlyStopping(logging,patience=args.patience, delta=0)
    for epoch in range(args.num_epochs):
        model.train()
        all_train_loss = 0
        for batch in train_dataloader:
            idxes, labels = [b.to(device) for b in batch]
            predictions = model(TrajPOIGraph,padded_data)
            train_predictions = predictions[idxes]
            train_loss = F.cross_entropy(train_predictions, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            all_train_loss += train_loss.item()

        with torch.no_grad():
            model.eval()
            predictions = model(TrajPOIGraph,padded_data)
            val_predictions = predictions[init_dataset.val_idx]
            val_label = torch.tensor(init_dataset.val_uid_list, dtype=torch.long).to(device)
            # val_predictions = predictions[len(init_dataset.train_uid_list):len(init_dataset.train_uid_list) + len(init_dataset.val_uid_list)]
            val_loss = F.cross_entropy(val_predictions, val_label)
            val_acc_list = accuracy(val_predictions, val_label, topk=(1, 5, 10,))

            train_predictions = predictions[init_dataset.train_idx]
            train_label = torch.tensor(init_dataset.train_uid_list, dtype=torch.long).to(device)
            train_acc_list = accuracy(train_predictions, train_label, topk=(1, 5, 10,))
            if early_stopping(val_loss.item()):
                logging.info("Early stopping triggered!")
                break
            if val_acc_list[0] > best_acc:
                best_model = copy.deepcopy(model)
                best_acc = val_acc_list[0]
        scheduler.step(val_loss)
        logging.info("Epoch: {}, TrainingLoss: {:.4f}, ValidationLoss: {:.4f}".format(epoch, all_train_loss, val_loss))
        logging.info(
            "Training: ACC@1: {:4f}, ACC@5: {:4f}, ACC@10: {:4f}, Macro@F: {:4f}, Macro@P: {:4f}, Macro@R: {:4f}".format(
                train_acc_list[0].item(), train_acc_list[1].item(), train_acc_list[2].item(), train_acc_list[3],
                train_acc_list[4], train_acc_list[5]))
        logging.info(
            "Validation: ACC@1: {:4f}, ACC@5: {:4f}, ACC@10: {:4f}, Macro@F: {:4f}, Macro@P: {:4f}, Macro@R: {:4f}".format(
                val_acc_list[0].item(), val_acc_list[1].item(), val_acc_list[2].item(), val_acc_list[3],
                val_acc_list[4], val_acc_list[5]))

        with torch.no_grad():
            model.eval()
            predictions = model(TrajPOIGraph,padded_data)
            test_label = torch.tensor(init_dataset.test_uid_list, dtype=torch.long).to(device)
            test_predictions = predictions[init_dataset.test_idx]
            acc_list = accuracy(test_predictions, test_label, topk=(1, 5, 10,))
            logging.info("ACC@1: {:4f}, ACC@5: {:4f}, ACC@10: {:4f}, Macro@F: {:4f}, Macro@P: {:4f}, Macro@R: {:4f}".format(
                acc_list[0].item(), acc_list[1].item(), acc_list[2].item(), acc_list[3], acc_list[4], acc_list[5]))

    logging.info("7. Final Results")
    Globalmodel = best_model.to(device)
    test_result = pd.DataFrame()
    with torch.no_grad():
        Globalmodel.eval()
        predictions = Globalmodel(TrajPOIGraph,padded_data)
        test_label = torch.tensor(init_dataset.test_uid_list, dtype=torch.long).to(device)
        test_predictions = predictions[init_dataset.test_idx]
        test_result['trajs'] = init_dataset.test_trajs
        test_result['labels'] = test_label.cpu().numpy()
        test_result['predictions'] = test_predictions.argmax(-1).squeeze().cpu().numpy()
        test_result.to_csv('testresult.csv',sep=';')
        acc_list = accuracy(test_predictions, test_label, topk=(1, 5, 10, 20))
        logging.info("{}time/10,ACC@1: {:4f}, ACC@5: {:4f}, ACC@10: {:4f},ACC@20: {:4f}, Macro@F: {:4f}, Macro@P: {:4f}, Macro@R: {:4f}".format(
            repeat_idx,acc_list[0].item(), acc_list[1].item(), acc_list[2].item(), acc_list[3].item(), acc_list[4], acc_list[5], acc_list[6]))

        for label_class, idx_list in init_dataset.testclass_idx.items():
            # 获取对应类别的标签和预测
            class_labels = test_label[idx_list]
            class_predictions = test_predictions[idx_list]
            # 计算该类别的准确率
            acc_list = accuracy(class_predictions, class_labels, topk=(1, 5, 10,20))
            logging.info("usertype:{},{}time/10,ACC@1: {:4f}, ACC@5: {:4f}, ACC@10: {:4f},ACC@20: {:4f}, Macro@F: {:4f}, Macro@P: {:4f}, Macro@R: {:4f}".format(
            label_class,repeat_idx,acc_list[0].item(), acc_list[1].item(), acc_list[2].item(), acc_list[3].item(), acc_list[4], acc_list[5], acc_list[6]))

