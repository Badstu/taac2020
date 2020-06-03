import pandas as pd
import os
import numpy as np

train_ad_path = 'work/train_preliminary/ad.csv'
train_click_log_path = 'work/train_preliminary/click_log.csv'

train_ad = pd.read_csv(train_ad_path, index_col=False)
train_click_log = pd.read_csv(train_click_log_path, index_col=False)
print(train_ad.shape)
print(train_click_log.shape)

# replace "\\N"  with "0"
train_ad = train_ad.replace("\\N", "0")
train_click_log = train_click_log.replace("\\N", "0")

train_ad.dropna(axis=0,how='any', inplace=True) 
train_click_log.dropna(axis=0,how='any', inplace=True) 
print(train_ad.shape)
print(train_click_log.shape)

# merge & sort
ad_click = train_click_log.merge(train_ad, on='creative_id')
ad_click.sort_values(by=['user_id', 'time'], inplace=True)
print('merge shape:\t', ad_click.shape)

# change type
ad_click['product_id'] = ad_click['product_id'].astype(int)
ad_click['ad_id'] = ad_click['ad_id'].astype(int)
ad_click['advertiser_id'] = ad_click['advertiser_id'].astype(int)

## 对于每位user, 将 91天内点击过的 product_id 整合为一个列表, 
## groupby保留每个组内的行顺序
user_pr_id = ad_click.groupby('user_id')['product_id'].apply(np.array).reset_index(name='product_id')
print('groupby user_pr_id!')
## 对于每位user, 将 91天内点击过的 ad_id 整合为一个列表, groupby保留每个组内的行顺序
user_ad_id = ad_click.groupby('user_id')['ad_id'].apply(np.array).reset_index(name='ad_id')
print('groupby user_ad_id!')
## 对于每位user, 将 91天内点击过的 advertiser_id 整合为一个列表, groupby保留每个组内的行顺序
user_adser_id = ad_click.groupby('user_id')['advertiser_id'].apply(np.array).reset_index(name='advertiser_id')
print('groupby user_adser_id!')

# 使用分位数减少用户数量
# 删除上下分位数以外的 user
lower = 0.0025
upper = 0.9975
user_pr_id['count_click'] = [len(x) for x in user_pr_id['product_id']]
quantiles = user_pr_id['count_click'].quantile([lower, 0.5, upper])
print(f'quaniles[{lower}, 0.5, {upper}]:\t', quantiles)

user_pr_id['whether_remain'] = [(quantiles[lower] <= c <= quantiles[upper]) for c in user_pr_id['count_click']]

# drop users whose "whether_remain" is False
reduce_pr_id = user_pr_id[user_pr_id['whether_remain'].isin([True])].reset_index(drop=True)
print('reduce pr id:\t', reduce_pr_id.shape)

user_ad_id['whether_remain'] = [(quantiles[lower] <= c <= quantiles[upper]) for c in user_pr_id['count_click']]
reduce_ad_id = user_ad_id[user_ad_id['whether_remain'].isin([True])].reset_index(drop=True)
print('reduce ad id:\t', reduce_ad_id.shape)

user_adser_id['whether_remain'] = [(quantiles[lower] <= c <= quantiles[upper]) for c in user_pr_id['count_click']]
reduce_adser_id = user_adser_id[user_adser_id['whether_remain'].isin([True])].reset_index(drop=True)
print('reduce ad id:\t', reduce_adser_id.shape)

np.save('work/train/reduce_user_id.npy', reduce_pr_id['user_id'].to_numpy())
np.save('work/train/reduce_pr_id.npy', reduce_pr_id['product_id'].to_numpy())
np.save('work/train/reduce_ad_id.npy', reduce_ad_id['ad_id'].to_numpy())
np.save('work/train/reduce_advertiser_id.npy', reduce_adser_id['advertiser_id'].to_numpy())


# # drop duplicate
# reduce_pr_id['pr_id_unique'] = [np.unique(x) for x in reduce_pr_id['product_id']]
# reduce_ad_id['ad_id_unique'] = [np.unique(x) for x in reduce_ad_id['ad_id']]
# reduce_adser_id['adser_id_unique'] = [np.unique(x) for x in reduce_adser_id['advertiser_id']]

# reduce_pr_id.to_csv('work/train/reduce_pr_id.csv')
# reduce_ad_id.to_csv('work/train/reduce_ad_id.csv')
# reduce_adser_id.to_csv('work/train/reduce_adser_id.csv')
