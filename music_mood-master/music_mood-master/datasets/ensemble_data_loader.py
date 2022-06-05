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

from .custom_data_loader import CustomDataLoader


class EnsembleLoader(CustomDataLoader):
    def __init__(self, config):
        """
        :param config:
        """
        super().__init__(config)

    def set_data_loader(self):
        self.config['train_col'] = self.config['train_col_A']
        self.config['data_mode'] = 'ensemble'
        X_B, y_B, z_B = self.get_data('B')
        self.X, self.y, self.z = X_B, y_B, z_B
        X_train_B, _y_train_B, X_val_B, _y_val_B, X_test_B, y_test_B = self.get_split_data_mp3()

        df_A_combined = self.get_data('A')
        df_A_combined = self.remove_duplicates(df_A_combined)
        df_A_combined['mood'].replace(utils.class2idx, inplace=True)
        df_A_combined = df_A_combined[df_A_combined['id'].isin(z_B)]
        df_A_combined = df_A_combined.sort_values(by=[self.config['predict_col'], 'id'])
        df_A = self.extract_cols_from_df(df_A_combined, self.config['train_col'], self.config['predict_col'])
        self.df = df_A
        z_A = df_A_combined['id']
        X_A, y_A = self.get_x_y(df_A)
        self.X, self.y, self.z = X_A, y_A, z_A
        X_train_A, _y_train_A, X_val_A, _y_val_A, X_test_A, y_test_A = self.get_split_data()
        self.X_train_B = X_train_B

        # X_train, self._y_train, X_val, self._y_val, X_test, self.y_test = self.get_split_data()

        X_train_A, _y_train_A, X_val_A, _y_val_A, X_test_A, y_test_A = self.normalize_all_data(X_train_A, _y_train_A, X_val_A, _y_val_A, X_test_A, y_test_A, 'A')
        X_train_B, _y_train_B, X_val_B, _y_val_B, X_test_B, y_test_B = self.normalize_all_data(X_train_B, _y_train_B, X_val_B, _y_val_B, X_test_B, y_test_B, 'B')
        self.y_test_A,  self._y_train_A, self._y_val_A = y_test_A, _y_train_A, _y_val_A
        self.train_loader_A, self.val_loader_A, self.test_loader_A = self.create_data_loader(X_train_A, _y_train_A,
                                                                                             X_val_A, _y_val_A,
                                                                                             X_test_A, y_test_A)
        self.train_loader_B, self.val_loader_B, self.test_loader_B = self.create_data_loader(X_train_B, _y_train_B,
                                                                                             X_val_B, _y_val_B,
                                                                                             X_test_B, y_test_B)

        # X_train, self._y_train, X_val, self._y_val, X_test, self.y_test = self.normalize_all_data(X_train,
        #                                                                                           self._y_train, X_val,
        #                                                                                           self._y_val, X_test,
        #                                                                                           self.y_test)
        # self.train_loader, self.val_loader, self.test_loader = self.create_data_loader(X_train, self._y_train, X_val,
        #                                                                                self._y_val, X_test, self.y_test)

    def get_data(self, model_name):
        if self.config['data_mode_' + model_name] == "mp3":
            X, y, ids = self.load_mp3s()
            return X, y, ids
        elif self.config['data_mode_' + model_name] == "csv":
            df_combined = self.load_csvs()

            col = ['id']
            col += self.config['train_col_' + model_name]

            df_combined = self.extract_cols_from_df(df_combined, col, self.config['predict_col'])
            return df_combined

    def normalize_all_data(self, X_train, y_train, X_val, y_val, X_test, y_test, data_loader_type=None):

        if data_loader_type == 'A':
            X_train_A, X_val_A, X_test_A = self.normalize_data('A', X_train, X_val, X_test)
            X_train = X_train_A
            X_val = X_val_A
            X_test = X_test_A
        elif data_loader_type == 'B':
            X_train_B, X_val_B, X_test_B = self.normalize_data('B', X_train, X_val, X_test)
            X_train = X_train_B
            X_val = X_val_B
            X_test = X_test_B
        else:
            X_train_A, X_val_A, X_test_A = self.normalize_data('A', X_train, X_val, X_test)
            X_train_B, X_val_B, X_test_B = self.normalize_data('B', X_train, X_val, X_test)

            X_train = np.concatenate((X_train_A, X_train_B), axis=1)
            X_val = np.concatenate((X_val_A, X_val_B), axis=1)
            X_test = np.concatenate((X_test_A, X_test_B), axis=1)

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_val, y_val = np.array(X_val), np.array(y_val)
        X_test, y_test = np.array(X_test), np.array(y_test)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def normalize_data(self, model_name, X_train, X_val, X_test):
        cols = self.config['train_col_' + model_name]
        if model_name == 'A':
            X_train_model = X_train[cols]
            X_val_model = X_val[cols]
            X_test_model = X_test[cols]
        else:
            X_train_model = X_train
            X_val_model = X_val
            X_test_model = X_test

        scaler = MinMaxScaler()
        # Normalize Input
        if self.config['data_mode_' + model_name] == "mp3":
            X_train, X_val, X_test = self.normalize_mp3(scaler, X_train_model, X_val_model, X_test_model)
        elif self.config['data_mode_' + model_name] == "csv":
            X_train, X_val, X_test = self.normalize_csv(scaler, X_train_model, X_val_model, X_test_model)
        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

        return X_train, X_val, X_test

    def normalize_mp3(self, scaler, X_train, X_val, X_test):
        X_train = np.array(X_train).swapaxes(1, 2)
        X_val = np.array(X_val).swapaxes(1, 2)
        X_test = np.array(X_test).swapaxes(1, 2)
        X_train_list = []
        X_val_list = []
        X_test_list = []

        for i in X_train:
            X_train_list.append(scaler.fit_transform(i.reshape(-1, 1)).reshape(1, -1)[0])
        for i in X_val:
            X_val_list.append(scaler.transform(i.reshape(-1, 1)).reshape(1, -1)[0])
        for i in X_test:
            X_test_list.append(scaler.transform(i.reshape(-1, 1)).reshape(1, -1)[0])

        return X_train, X_val, X_test

    def normalize_csv(self, scaler, X_train, X_val, X_test):
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        return X_train, X_val, X_test

    def save_df_bar_img(self):
         labeled_df = self.df['mood'].replace(utils.idx2class, inplace=False)
         bar = labeled_df.value_counts().plot.bar()
         bar.figure.savefig('./figures/data_dist_' + self.config['data_mode_combined'] + '.jpeg')
         bar.figure.clear(True)

    def save_df_pie_img(self):
        labeled_df = self.df['mood'].replace(utils.idx2class, inplace=False)
        pie = labeled_df.value_counts().plot.pie(autopct='%.2f')
        pie.figure.savefig('./figures/data_dist_pie_' + self.config['data_mode_combined'] + '.jpeg')
        pie.figure.clear(True)

    def get_class_distribution(self, obj):
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
        train_count_df = pd.DataFrame.from_dict([self.get_class_distribution(self._y_train_A)]).melt()
        validation_count_df = pd.DataFrame.from_dict([self.get_class_distribution(self._y_val_A)]).melt()
        test_count_df = pd.DataFrame.from_dict([self.get_class_distribution(self.y_test_A)]).melt()
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

        fig.savefig('./figures/train_val_test_dist_' + self.config['data_mode_combined'] + '.jpeg')
        fig.clear(True)

    def finalize(self):
        self.save_df_pie_img()
        self.save_df_bar_img()
        self.plot_dist()