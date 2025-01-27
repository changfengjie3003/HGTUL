from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRAJS_PATH = 'raw_data/GOWALLA-1000-o/train_trajs.txt'
TRAJS_TIME_PATH = 'raw_data/GOWALLA-1000-o/train_trajs_time.txt'
NUM_USERS = 1000
uid_list = []
trajs_list = []
trajs_time_list = []
with open(TRAJS_PATH) as f:
    for traj in f:
        traj = traj.strip().split(" ")
        uid_list.append(int(traj[0]))
        trajs_list.append(list(map(int, traj[1:])))
with open(TRAJS_TIME_PATH) as f:
    for traj_time in f:
        traj_time = traj_time.strip().split(" ")
        trajs_time_list.append(list(map(int, traj_time)))
df = pd.DataFrame()
df['uid'] = uid_list
df['trajs'] = trajs_list
df['trajs_time'] = trajs_time_list
users = list(set(uid_list))
average_len = int(len(uid_list)/NUM_USERS)
delta = int(average_len*0.5)
appenddf_all = pd.DataFrame() 
for user in users:
    fre_user = uid_list.count(user)
    if average_len<=fre_user<=average_len+delta:
        continue
    elif fre_user>average_len+delta:
        droprate = round((fre_user-average_len-delta)/fre_user,2)
        udf = df.loc[df['uid']==user]
        sampled_indices = udf.sample(frac=droprate, random_state=42).index
        # 删除这些索引所在的行
        df = df.drop(sampled_indices)
        print(df.shape[0])
    else:
        repeatcounts = average_len-fre_user
        udf = df.loc[df['uid']==user]
        num_repeat_epoch = repeatcounts//fre_user
        appenddf = pd.DataFrame()
        if num_repeat_epoch != 0:
            for i in range(num_repeat_epoch):
                appenddf = appenddf.append(udf)
        repeatrate = round((repeatcounts%fre_user)/fre_user,2)
        sampled_df = udf.sample(frac=repeatrate, random_state=42)
        appenddf = appenddf.append(sampled_df)
        appenddf_all = appenddf_all.append(appenddf)
        appenddf_all = appenddf_all.reset_index(drop=True)
        print(appenddf)
df = df.append(appenddf_all)
df = df.sort_values(['uid'])
newtrajs_list = df['trajs']
newuid_list = df['uid']
newtrajs_time_list = df['trajs_time']
TRAJS_PATH2 = 'raw_data/GOWALLA-1000-2/train_trajs.txt'
TRAJS_TIME_PATH2 = 'raw_data/GOWALLA-1000-2/train_trajs_time.txt'
with open(TRAJS_PATH2, 'w') as f:
    for row in df.itertuples(index=False, name=None):
        usrid = row[0]
        path = row[1]
        f.write(str(usrid)+" ")
        f.write(" ".join(str(loc) for loc in path) + "\n")
with open(TRAJS_TIME_PATH2, 'w') as f:
    for row in df.itertuples(index=False, name=None):
        tlist = row[2]
        # 使用正则表达式匹配所有的日期时间
        # tlist_obj = eval(tlist, {'datetime': datetime})
        # tlist_str = [dt.strftime("%Y%m%d%H%M%S") for dt in tlist_obj]
        f.write(" ".join(str(time) for time in tlist) + "\n")
