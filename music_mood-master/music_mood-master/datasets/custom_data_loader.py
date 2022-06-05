import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# mp3 imports
import glob
import librosa


import utils.utils as utils
from .custom_dataset import ClassifierDataset

class CustomDataLoader():
    def __init__(self, config):
        """
        :param config:
        """
        self.config = config


    def set_data_loader(self):
        if self.config['data_mode'] == "mp3":
            self.X, self.y, self.z = self.load_mp3s()
            #hacky cleaning of Z (ids)
            # self.z = np.array([s.split('\\')[1].split('.mp3')[0] for s in self.z])
            self.df = pd.DataFrame(self.y, columns=['mood'])
            X_train, self._y_train, X_val, self._y_val, X_test, self.y_test = self.get_split_data_mp3()
        elif self.config['data_mode'] == "csv":
            self.df_combined = self.load_csvs()
            self.prepare_data()
            X_train, self._y_train, X_val, self._y_val, X_test, self.y_test = self.get_split_data()
        else:
            raise Exception("Please specify in the json a specified mode in data_mode")
        X_train, self._y_train, X_val, self._y_val, X_test, self.y_test = self.normalize_data(X_train, self._y_train, X_val, self._y_val, X_test, self.y_test)
        self.train_loader, self.val_loader, self.test_loader = self.create_data_loader(X_train, self._y_train, X_val, self._y_val, X_test, self.y_test)

    def prepare_data(self):
        self.df_combined = self.remove_duplicates(self.df_combined)
        self.df = self.extract_cols_from_df(self.df_combined, self.config['train_col'], self.config['predict_col'])
        self.z = self.df_combined['id']
        self.df['mood'].replace(utils.class2idx, inplace=True)
        self.X, self.y = self.get_x_y(self.df)

        # # Split into train+val and test
        # X_trainval, X_test, y_trainval, y_test = self.split_data(self.X, self.y, self.config['test_size_percent'])

        # # Split train into train-val
        # X_train, X_val, y_train, y_val = self.split_data(X_trainval, y_trainval, self.config['validation_size_percent'])

        # return X_train, y_train, X_val, y_val, X_test, y_test

    def get_split_data(self):
        self.X.reset_index(drop=True ,inplace=True)
        self.y.reset_index(drop=True ,inplace=True)
        self.z.reset_index(drop=True ,inplace=True)

        mood_train, mood_test_val, song_train, song_test_val, id_train, id_test_val = self.split_data_idx(self.y, self.z, self.config['train_size_percent'])
        X_train = self.X.loc[id_train]
        X_test_val = self.X.loc[id_test_val]
        y_train = self.y.loc[id_train]
        y_test_val = self.y.loc[id_test_val]
        # Split into train+val and test
        # X_trainval, X_test, y_trainval, self.y_test, ids_trainval, ids_test = self.split_data(self.X, self.y, self.z, self.config['test_size_percent'])
        mood_val, mood_test, song_val, song_test, id_val, id_test = self.split_data_idx(mood_test_val, song_test_val, self.config['validation_size_percent'])
        X_val = self.X.loc[id_val]
        X_test = self.X.loc[id_test]
        y_val = self.y.loc[id_val]
        y_test = self.y.loc[id_test]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_split_data_mp3(self):
        z = pd.Series(self.z)
        mood_train, mood_test_val, song_train, song_test_val, id_train, id_test_val = self.split_data_idx(self.y, z, self.config['train_size_percent'])
        X_train = self.X[id_train]
        X_test_val = self.X[id_test_val]
        y_train = self.y[id_train]
        y_test_val = self.y[id_test_val]
        # Split into train+val and test
        # X_trainval, X_test, y_trainval, self.y_test, ids_trainval, ids_test = self.split_data(self.X, self.y, self.z, self.config['test_size_percent'])
        mood_val, mood_test, song_val, song_test, id_val, id_test = self.split_data_idx(mood_test_val, song_test_val, self.config['validation_size_percent'])
        X_val = self.X[id_val]
        X_test = self.X[id_test]
        y_val = self.y[id_val]
        y_test = self.y[id_test]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def create_data_loader(self, X_train, y_train, X_val, y_val, X_test, y_test):
        train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
        test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

        TRAIN_BATCH_SIZE = self.config['batch_size']
        VAL_BATCH_SIZE = 1
        TEST_BATCH_SIZE = 1

        # Create instance from the loss
        if self.config['imbalance']:
            # obtain the count of all classes in our training set.
            self.class_weights = self.get_class_weight(y_train)
            weighted_sampler = self.get_weighted_random_sampler(train_dataset, self.class_weights)
            train_loader = DataLoader(dataset=train_dataset,
                          batch_size=TRAIN_BATCH_SIZE,
                          sampler=weighted_sampler
            )
        else:
            train_loader = DataLoader(dataset=train_dataset,
                          batch_size=TRAIN_BATCH_SIZE,
            )

        val_loader = DataLoader(dataset=val_dataset, batch_size=VAL_BATCH_SIZE)
        test_loader = DataLoader(dataset=test_dataset, batch_size=TEST_BATCH_SIZE)

        return train_loader, val_loader, test_loader


    def normalize_data(self, X_train, y_train, X_val, y_val, X_test, y_test):
        scaler = MinMaxScaler()

        # Normalize Input
        if self.config['data_mode'] == "mp3":
            X_train, y_train, X_val, y_val, X_test, y_test = self.normalize_mp3(scaler, X_train, y_train, X_val, y_val, X_test, y_test)
        elif self.config['data_mode'] == "csv":
            X_train, y_train, X_val, y_val, X_test, y_test = self.normalize_csv(scaler, X_train, y_train, X_val, y_val, X_test, y_test)
        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

        return  X_train, y_train, X_val, y_val, X_test, y_test

    def normalize_mp3(self, scaler, X_train, y_train, X_val, y_val, X_test, y_test):
        X_train = np.array(X_train).swapaxes(1,2)
        X_val = np.array(X_val).swapaxes(1,2)
        X_test = np.array(X_test).swapaxes(1,2)
        X_train_list = []
        X_val_list = []
        X_test_list = []

        for i in X_train:
            X_train_list.append(scaler.fit_transform(i))
        for i in X_val:
            X_val_list.append(scaler.transform(i))
        for i in X_test:
            X_test_list.append(scaler.transform(i))

        X_train, y_train = np.asarray(X_train_list), np.asarray(y_train)
        X_val, y_val = np.asarray(X_val_list), np.asarray(y_val)
        X_test, y_test = np.asarray(X_test_list), np.asarray(y_test)
        # X_train, self._y_train = np.asarray(X_train_list)[:, np.newaxis], np.asarray(self._y_train)[:, np.newaxis]
        # X_val, self._y_val = np.asarray(X_val_list)[:, np.newaxis], np.asarray(self._y_val)[:, np.newaxis]
        # X_test, self.y_test = np.asarray(X_test_list)[:, np.newaxis], np.asarray(self.y_test)[:, np.newaxis]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def normalize_csv(self, scaler, X_train, y_train, X_val, y_val, X_test, y_test):
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_val, y_val = np.array(X_val), np.array(y_val)
        X_test, y_test = np.array(X_test), np.array(y_test)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def load_csvs(self):
        moods = self.config['moods']

        files = []
        for g in moods:
            for filename in os.listdir(f'./data/{g}'):
                file_name = f'./data/{g}/{filename}'
                files.append(file_name)

        df = pd.concat([pd.read_csv(file) for file in files])
        return df

    def load_mp3s(self):
        X = np.load("./data/mp3/mp3features_s_m_t.npy")
        y = np.load("./data/mp3/mp3mood_s.npy")
        ids = np.load("./data/mp3/mp3song_s.npy")
        return X, y, ids

    def extract_cols_from_df(self, df, train_col, predict_col):
        df = df[train_col + [predict_col]]
        return df

    def get_x_y(self, df):
        X = df.iloc[:, 0:-1]
        y = df.iloc[:, -1]
        return X, y

    def split_data(self, X, y, z, size):
        return train_test_split(X, y, z, test_size=size, stratify=y, random_state=self.config["seed"])

    def split_data_idx(self, y, z, size):
        return train_test_split(y, z, z.index, stratify=y,  test_size=size, random_state=self.config["seed"])


    def get_weighted_random_sampler(self, train_dataset, class_weight):
        target_list = []
        for _, t in train_dataset:
            target_list.append(t)

        target_list = torch.tensor(target_list)
        target_list = target_list[torch.randperm(len(target_list))]

        class_weights_all = class_weight[target_list]

        # initialize our WeightedRandomSampler
        weighted_sampler = WeightedRandomSampler(
            weights=class_weights_all,
            num_samples=len(class_weights_all),
            replacement=True
        )

        return weighted_sampler

    def get_class_weight(self, y_train):
        class_count = [i for i in self.get_class_distribution(y_train).values()]
        class_weights = 1./torch.tensor(class_count, dtype=torch.float)
        return class_weights

    def get_df_shape(self):
        return self.df.shape


    def remove_duplicates(self, df):
        # making a bool series
        bool_series = df["id"].duplicated()
        df[bool_series].to_csv(r'./data/duplicates_' + self.config['data_mode'] + '.csv', index = False)
        return df[~bool_series]

    def save_df_bar_img(self):
        labeled_df = self.df['mood'].replace(utils.idx2class, inplace=False)
        bar = labeled_df.value_counts().plot.bar()
        bar.figure.savefig('./figures/data_dist_' + self.config['data_mode']  + '_' + self.config["model"] + '.jpeg')
        bar.figure.clear(True)

    def save_df_pie_img(self):
        labeled_df = self.df['mood'].replace(utils.idx2class, inplace=False)
        pie = labeled_df.value_counts().plot.pie(autopct='%.2f')
        pie.figure.savefig('./figures/data_dist_pie_' + self.config['data_mode']  + '_' + self.config["model"] + '.jpeg')
        pie.figure.clear(True)

    def get_class_distribution(self,obj):
        # Visualize Class Distribution in Train, Val, and Test
        count_dict = {
            "calm": 0,
            "energetic": 0,
            "happy": 0,
            "sad": 0
        }

        for i in obj:
            if i == 0:
                count_dict['calm'] += 1
            elif i == 1:
                count_dict['energetic'] += 1
            elif i == 2:
                count_dict['happy'] += 1
            elif i == 3:
                count_dict['sad'] += 1
            else:
                print("Check classes.")

        return count_dict

    def plot_dist(self):
        train_count_df =  pd.DataFrame.from_dict([self.get_class_distribution(self._y_train)]).melt()
        validation_count_df =  pd.DataFrame.from_dict([self.get_class_distribution(self._y_val)]).melt()
        test_count_df =  pd.DataFrame.from_dict([self.get_class_distribution(self.y_test)]).melt()
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25,7))
        # Train
        axes[0].bar(train_count_df["variable"], train_count_df["value"])
        axes[0].set_title('Class Distribution in Train Set')

        # Validation
        axes[1].bar(validation_count_df["variable"], validation_count_df["value"])
        axes[1].set_title('Class Distribution in Validation Set')

        # Test
        axes[2].bar(test_count_df["variable"], test_count_df["value"])
        axes[2].set_title('Class Distribution in Test Set')

        fig.savefig('./figures/train_val_test_dist_' + self.config['data_mode']  + '_' + self.config["model"] + '.jpeg')
        fig.clear(True)

    def finalize(self):
        self.save_df_pie_img()
        self.save_df_bar_img()
        self.plot_dist()