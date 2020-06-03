from keras.utils import Sequence
from keras.utils import to_categorical

import numpy as np

class DataGenerator(Sequence):

    def __init__(self, file_list, label_file_list, batch_size=8, dim=(8, 160), n_channels=3,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.file_list = file_list
        self.label_file_list = label_file_list
        self.batch_size = batch_size
        self.dim = dim # (8, 160): 8: 一个npy文件里的user个数, 160: 特征向量长度
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle

        # 3 input
        assert len(file_list) == 3
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        assert len(self.file_list[0]) == len(self.file_list[1]) == len(self.file_list[2])
        return int(np.floor(len(self.file_list[0]) / self.batch_size))

    def __getitem__(self, index):
        'Get next batch'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        file_list_temp_0 = [self.file_list[0][k] for k in indexes]
        file_list_temp_1 = [self.file_list[1][k] for k in indexes]
        file_list_temp_2 = [self.file_list[2][k] for k in indexes]
        file_list_temp = [file_list_temp_0, file_list_temp_1, file_list_temp_2]
        label_file_temp = [self.label_file_list[k] for k in indexes]
       
        # Array of X_train and y_train
        X, y = self.__data_generation(file_list_temp, label_file_temp)
        return [X[0], X[1], X[2]], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.file_list[0]))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, file_list_temp, label_file_temp):
        'Generates data containing batch_size samples'
        # X: (n_samples, *dim)
        # y: (n_samples, )
        # Initialization
        X = np.empty((self.n_channels, self.batch_size*self.dim[0], self.dim[1]))
        y = np.empty((self.batch_size*self.dim[0]), dtype=int)
        assert self.batch_size == len(file_list_temp[0])
        
        for i in range(self.n_channels):
            for idx in range(self.batch_size):
                # Store sample
                x_file_path = file_list_temp[i][idx]
                X[i,idx*self.dim[0]:(idx+1)*self.dim[0] ] = np.load(x_file_path)

                # Store class
                if i == 0:
                    y_file_path = label_file_temp[idx]
                    y[idx*self.dim[0]:(idx+1)*self.dim[0]] = np.load(y_file_path)
            
        return X, to_categorical(y-1, num_classes=self.n_classes)
