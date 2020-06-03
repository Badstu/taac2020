
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence

import glob
import os
import numpy as np
import pandas as pd

from data_generator import DataGenerator
from model import get_model

nb_class = 10

max_click_num = np.load('work/train/mask/mask_ad_id/ad_id_0000.npy').shape[1]
input_length_list = [max_click_num, max_click_num, max_click_num]

index_ad = 3812196  # ad_id 中最大的数
index_pr = 43157    # product_id 中最大的数
index_adser = 62958 # advertiser_id 中最大的数
index_list = [index_ad+1, index_pr+1, index_adser+1]    

embedding_matrix = None # 权重向量为None
model = get_model(index_list, nb_class, 100, input_length_list, embedding_matrix)


# ====================
# train set
# ====================
mask_pr_id_path = 'work/train/mask/mask_pr_id/*.npy'
mask_pr_id = sorted(glob.glob(mask_pr_id_path))
mask_ad_id_path = 'work/train/mask/mask_ad_id/*.npy'
mask_ad_id = sorted(glob.glob(mask_ad_id_path))
mask_adser_id_path = 'work/train/mask/mask_adser_id/*.npy'
mask_adser_id = sorted(glob.glob(mask_adser_id_path))
age_path = 'work/train/user_age/*.npy'
age = sorted(glob.glob(age_path),)

# 用一半的数据集试一试
print(len(mask_pr_id) ) 
num_npy = int(np.floor(len(mask_pr_id) / 2))
# split: 7:3
split = int(0.7 * num_npy)

batch_size = 8
partition = [mask_ad_id[0:split], mask_pr_id[0:split], mask_adser_id[0:split]]
val_partition = [mask_ad_id[split:num_npy], mask_pr_id[split:num_npy], mask_adser_id[split:num_npy]]

partition_label = age[0:split]
val_partition_label = age[split:num_npy]

def change_dataset_size(x, y, batch_size):   
    length = len(x)   
    if (length % batch_size != 0):       
        remainder = length % batch_size       
        x = x[:(length - remainder)]       
        y = y[:(length - remainder)]    
    return x, y

for i in range(len(partition)):
    partition[i], partition_label = change_dataset_size(partition[i], partition_label, batch_size)
    val_partition[i], val_partition_label = change_dataset_size(val_partition[i], val_partition_label, batch_size)

# hyperparameters
params = {'dim': (8, 160),
          'batch_size': batch_size,
          'n_classes': 10,
          'n_channels': 3,
          'shuffle': True}

# Generator train set & valid data
training_generator = DataGenerator(partition, partition_label, **params)
validation_generator = DataGenerator(val_partition, val_partition_label, **params) # work same as training generator

# print(len(training_generator))
hst = model.fit_generator(generator=training_generator,
                           epochs=20,
                           validation_data=validation_generator,
                           steps_per_epoch = int(np.floor(num_npy / batch_size)),
                           verbose = 1,
                           use_multiprocessing=False,
                           max_queue_size=32)
# Data/train/user_age/user_age0.npy

