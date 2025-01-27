import torch
from datetime import datetime
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import geohash2
from collections import Counter
import math
import pandas as pd

class InitDataset:
    def __init__(self, args):
        self.args = args
        self.m = 1

        self.tz = None

        datasetname = args.dataset
        user_num = args.user_number
        self.datadir = f"raw_data/{datasetname}-{user_num}/"
        self.test_trajs_path = self.datadir+"test_trajs.txt"
        self.test_trajs_time_path = self.datadir+"test_trajs_time.txt"
        self.val_trajs_path = self.datadir+"val_trajs.txt"
        self.val_trajs_time_path = self.datadir+"val_trajs_time.txt"
        self.train_trajs_path = self.datadir+"train_trajs.txt"
        self.train_trajs_time_path = self.datadir+"train_trajs_time.txt"
        self.vidx_to_latlon_path = self.datadir+"vidx_to_latlon.txt"

        self.test_uid_list, self.test_trajs = self.read_trajs_file(self.test_trajs_path)
        self.test_timestamps_list, self.test_weekdays_list, self.test_hours_list = self.read_trajs_time_file(self.test_trajs_time_path)
        self.val_uid_list, self.val_trajs = self.read_trajs_file(self.val_trajs_path)
        self.val_timestamps_list, self.val_weekdays_list, self.val_hours_list = self.read_trajs_time_file(self.val_trajs_time_path)
        self.train_uid_list, self.train_trajs = self.read_trajs_file(self.train_trajs_path)
        self.train_timestamps_list, self.train_weekdays_list, self.train_hours_list = self.read_trajs_time_file(self.train_trajs_time_path)

        idxes = list(range(len(self.train_trajs)+len(self.val_trajs)+len(self.test_trajs)))
        self.train_idx = idxes[:len(self.train_trajs)]
        self.val_idx = idxes[len(self.train_trajs):len(self.train_trajs)+len(self.val_trajs)]
        self.test_idx = idxes[-len(self.test_trajs):]
        self.n_all_trajs = len(self.train_trajs)+len(self.val_trajs)+len(self.test_trajs)

        # latlon for constructing Ngraph
        self.vidx_to_latlon,self.vidx_to_geoid = self.read_vidx_to_latlon(self.vidx_to_latlon_path)

        # user set for linking
        self.users = dict([(uid, idx) for idx, uid in enumerate(set(self.train_uid_list))])

        # vocabs
        # self.vocabs = {"PAD":0, "UNK": 1}
        self.vocabs = {}
        self.build_vocabs(self.train_trajs, self.vocabs)
        self.build_vocabs(self.test_trajs, self.vocabs)
        self.build_vocabs(self.val_trajs, self.vocabs)

        self.train_uid_list = [self.users[uid] for uid in self.train_uid_list]
        self.val_uid_list = [self.users[uid] for uid in self.val_uid_list]
        self.test_uid_list = [self.users[uid] for uid in self.test_uid_list]
        self.geovocabs = self.geo_vocabs()
        self.vidx_to_geoid = self.strtogeoid()
        self.m_train_idx = self.obtain_training_idxes(self.m)

        self.act_label_path = f"activate_label/{datasetname}-{user_num}-o-ulabel.csv"
        self.user_act()

    def user_act(self):
        df =pd.read_csv(self.act_label_path,sep=';')
        usertype_dict = {}
        for i in range(df.shape[0]):
            usrid = self.users[df.loc[i,'userid']]
            usertype = df.loc[i,'activate_label']
            usertype_dict[usrid] = usertype
        testclass_idx = {'0':[],'1':[],'2':[]}
        for idx in range(len(self.test_uid_list)):
            type = usertype_dict[self.test_uid_list[idx]]
            if type == 0:
                testclass_idx["0"].append(idx)
            elif type == 1:
                testclass_idx['1'].append(idx)
            else:
                testclass_idx['2'].append(idx)
        self.testclass_idx = testclass_idx
        print('aa')

    
    def strtogeoid(self):
        vidx_to_geocode = dict()
        for key,value in self.vidx_to_geoid.items():
            vidx_to_geocode[key] = self.geovocabs[value]
        return vidx_to_geocode

    def obtain_training_trajs_by_m_train_idx(self):
        train_trajs, train_uid_list = [], []
        for idx in self.m_train_idx:
            train_trajs.append(self.train_trajs[idx])
            train_uid_list.append(self.train_uid_list[idx])
        return train_trajs, train_uid_list

    def obtain_training_idxes(self, m):
        assert 0 < m and m <= 1
        np.random.seed(0)
        train_idxs = list(range(len(self.train_trajs)))
        np.random.shuffle(train_idxs)
        train_end = int(self.m * len(self.train_trajs))
        idxes = train_idxs[:train_end]
        return idxes

    def build_vocabs(self, trajs, vocabs):
        for traj in trajs:
            for point in traj:
                if point not in vocabs:
                    vocabs[point] = len(vocabs)
    
    def geo_vocabs(self):
        geo_vocabs = dict()
        for value in self.vidx_to_geoid.values():
            if value not in geo_vocabs:
                geo_vocabs[value] = len(geo_vocabs)
        return geo_vocabs

    def read_trajs_file(self, filepath):
        uid_list = []
        trajs_list = []
        with open(filepath) as f:
            for traj in f:
                traj = traj.strip().split(" ")
                uid_list.append(int(traj[0]))
                trajs_list.append(list(map(int, traj[1:])))
        return uid_list, trajs_list

    def read_trajs_time_file(self, filepath):
        timestamps_list = []
        weekdays_list = []
        hours_list = []
        with open(filepath) as f:
            for traj_time in f:
                traj_time = traj_time.strip().split(" ")
                # 解析日期字符串为datetime对象
                datetime_timestamps = [datetime.strptime(timestamp, "%Y%m%d%H%M%S") for timestamp in traj_time]
                # 提取时间戳，转换为浮点型（如有必要可以修改为其它格式）
                timestamps = [dt.timestamp() for dt in datetime_timestamps]
                # 计算星期几
                weekdays = np.array([dt.weekday() for dt in datetime_timestamps])
                # 周一到周五标记为0，周末标记为1
                weekdays[weekdays < 5] = 0
                weekdays[weekdays > 4] = 1
                # 计算小时（每半小时作为一个时段）
                hours = np.array([dt.hour * 2 for dt in datetime_timestamps])
                # 添加到相应的列表中
                timestamps_list.append(timestamps)
                weekdays_list.append(weekdays.tolist())
                hours_list.append(hours.tolist())
        
        return timestamps_list, weekdays_list, hours_list

    # def read_trajs_time_file(self, filepath):
    #     timestamps_list = []
    #     weekdays_list = []
    #     hours_list = []
    #     with open(filepath) as f:
    #         for traj_time in f:
    #             traj_time = traj_time.strip().split(" ")
    #             timestamps = list(map(float, traj_time))
    #             datetime_timestamps = [datetime.fromtimestamp(timestamp, tz=self.tz) for timestamp in map(float, traj_time)]
    #             weekdays = np.array([timestamp.weekday() for timestamp in datetime_timestamps])
    #             # zero for week day, one for weekends
    #             weekdays[weekdays < 5] = 0
    #             weekdays[weekdays > 4] = 1
    #             # time slot is half hour
    #             hours = np.array([timestamp.hour*2 for timestamp in datetime_timestamps])
    #             timestamps_list.append(timestamps)
    #             weekdays_list.append(weekdays.tolist())
    #             hours_list.append(hours.tolist())
    #     return timestamps_list, weekdays_list, hours_list


    def read_vidx_to_latlon(self, filepath):
        vidx_to_latlon = dict()
        vidx_to_geoid = dict()
        with open(filepath) as f:
            for line in f:
                vidx, lat, lon = line.strip().split(" ")
                vidx_to_latlon[int(vidx)] = [float(lat), float(lon)]
                vidx_to_geoid[int(vidx)] = geohash2.encode(float(lat), float(lon))[0:7]
        return vidx_to_latlon,vidx_to_geoid

class BatchedDataset:
    def __init__(self, idxes, labels):
        self.idxes = idxes
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.idxes[i], self.labels[i]

class BatchedFusionDataset:
    def __init__(self, trajs, idxes, labels):
        self.trajs = trajs
        self.idxes = idxes
        self.labels = labels

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, i):
        return torch.tensor(self.trajs[i], dtype=torch.long), len(self.trajs[i]), self.idxes[i], self.labels[i]

    def collate_fun(self, data):
        trajs = [d[0] for d in data]
        trajs_len = [d[1] for d in data]
        idxes = [d[2] for d in data]
        labels = [d[3] for d in data]
        pad_trajs = pad_sequence(trajs, batch_first=True)
        return pad_trajs, torch.tensor(trajs_len, dtype=torch.long), torch.tensor(idxes, dtype=torch.long), \
               torch.tensor(labels, dtype=torch.long)