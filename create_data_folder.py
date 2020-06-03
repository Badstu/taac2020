import numpy as np
import shutil  
import os
import pandas as pd

ad_id = np.load('work/train/reduce_ad_id.npy', allow_pickle=True)
print(ad_id.shape)
pr_id = np.load('work/train/reduce_pr_id.npy', allow_pickle=True)
print(pr_id.shape)
adser_id = np.load('work/train/reduce_advertiser_id.npy', allow_pickle=True)
print(adser_id.shape)

# check 3 inputs
# 比较所有用户中, 91天内, 点击product_id的数量中的最大值
max_click_num = np.sort([a.shape[0] for a in pr_id])[-1]
print(max_click_num)
max_click_num_1 = np.sort([a.shape[0] for a in pr_id])[-1]
print(max_click_num_1)
max_click_num_2 = np.sort([a.shape[0] for a in adser_id])[-1]
assert max_click_num == max_click_num_1 == max_click_num_2

# find abnormal 
abnormal = np.where(np.array([a.shape[0] for a in pr_id]) > max_click_num)
num_abnormal = len(abnormal[0])
print('num_abnormal:\t', num_abnormal)
assert num_abnormal == 0
print('abnormal:\t', abnormal)

# 排除异常值
mask_pr_id = np.zeros([pr_id.shape[0]-num_abnormal, max_click_num])
mask_ad_id = np.zeros([ad_id.shape[0]-num_abnormal, max_click_num])
mask_adser_id = np.zeros([adser_id.shape[0]-num_abnormal, max_click_num])
print('after delete abnormal:\t', mask_pr_id.shape)

# 生成数据文件夹, 便于data generator取数据
shutil.rmtree('work/train/mask/mask_pr_id/')  
os.mkdir('work/train/mask/mask_pr_id/')  
shutil.rmtree('work/train/mask/mask_ad_id/')  
os.mkdir('work/train/mask/mask_ad_id/')  
shutil.rmtree('work/train/mask/mask_adser_id/')  
os.mkdir('work/train/mask/mask_adser_id/')  

# 生成数据文件夹, 便于data generator取数据
k = 0
inteval = 8
for i in range(mask_pr_id.shape[0]):
    if pr_id[i].shape[0] <= max_click_num:
        if i % inteval == 0 and i != 0:
            np.save(f'work/train/mask/mask_pr_id/pr_id_{str(k).zfill(4)}.npy',mask_pr_id[i-inteval:i])  
            np.save(f'work/train/mask/mask_ad_id/ad_id_{str(k).zfill(4)}.npy',mask_ad_id[i-inteval:i]) 
            np.save(f'work/train/mask/mask_adser_id/adser_id_{str(k).zfill(4)}.npy',mask_adser_id[i-inteval:i]) 
            k += 1
            print(k)
        mask_pr_id[i][0:pr_id[i].shape[0]] = pr_id[i]
        mask_ad_id[i][0:ad_id[i].shape[0]] = ad_id[i]
        mask_adser_id[i][0:ad_id[i].shape[0]] = adser_id[i]
    else: i -= 1
    
print('generate mask! ')

# print('train_user:')
train_user_path = 'work/train_preliminary/user.csv'
reduce_user_path = 'work/train/reduce_user_id.npy'

train_user = pd.read_csv(train_user_path, index_col=False)
reduce_user_id = np.load(reduce_user_path, allow_pickle=True)

reduce_user = train_user[train_user['user_id'].isin(reduce_user_id)].reset_index(drop=True).sort_values(by='user_id')
print(reduce_user.shape)

# 生成标签文件夹
shutil.rmtree('work/train/user_age/')  
os.mkdir('work/train/user_age/')  

shutil.rmtree('work/train/user_gender/')  
os.mkdir('work/train/user_gender/')  

# 保存文件, 便于从文件夹中读取数据, 5000是因为train的特征是5000个人一个npy
k=0
age = []
gender = []
for i in range(reduce_user.shape[0]):
    if i % inteval == 0 and i != 0:
        np.save(f'work/train/user_age/user_age_{str(k).zfill(4)}.npy', age)
        np.save(f'work/train/user_gender/user_gender_{str(k).zfill(4)}.npy', gender)
        k+=1
        age = []
        gender = []
        print(k)
    print(i)
    age.append(reduce_user['age'][i])
    gender.append(train_user['gender'][i])
