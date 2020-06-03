import numpy as np
import glob
import pandas as pd

train_user = pd.read_csv('work/train_preliminary/user.csv')
train_ad = pd.read_csv('work/train_preliminary/ad.csv')
train_click_log = pd.read_csv('work/train_preliminary/click_log.csv')

# print(train_click_log.info())
# replace rows with missing values "0"
train_ad = train_ad.replace("\\N", "0")
train_click_log = train_click_log.replace("\\N", "0")
print(train_ad.shape)
print(train_click_log.shape)

click = train_click_log.merge(train_ad, on='creative_id')
print(click.shape)
# click.dropna(axis=0,how='any',inplace=True)
# print(click.dropna(axis=0,how='any').shape)
# print(click.info())

## change type
click['product_id'] = click['product_id'].astype(int)
# print(click['product_id'])
click['ad_id'] = click['ad_id'].astype(int)
click['advertiser_id'] = click['advertiser_id'].astype(int)

reduce_user_id = np.load('work/train/reduce_user_id.npy', allow_pickle=True)
reduce_pr_id = np.load('work/train/reduce_pr_id.npy', allow_pickle=True)
reduce_ad_id = np.load('work/train/reduce_ad_id.npy', allow_pickle=True)
reduce_adser_id = np.load('work/train/reduce_advertiser_id.npy', allow_pickle=True)

# df = pd.DataFrame([reduce_user_id, reduce_ad_id, reduce_pr_id, reduce_adser_id], columns=['user_id', 'ad_id', 'pr_id', 'adser_id'], index=range(reduce_user_id.shape[0]))
for idx, uid in enumerate(reduce_user_id):
    if idx % 10000 == 0: 
        print(idx)
    df = click.loc[click['user_id']==uid]
    for aid in reduce_ad_id[idx]:
        # for t in df['ad_id']:
        if (aid) not in df['ad_id'].values:
            print('aid not match')
    for pid in reduce_pr_id[idx]:
        if (pid) not in df['product_id'].values:
            print('pid not match')
    for adserid in reduce_adser_id[idx]:
        if (adserid) not in df['advertiser_id'].values:
            print('adserid not match')
    


