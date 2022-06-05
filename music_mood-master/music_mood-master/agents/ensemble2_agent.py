"""
Main Agent for MyEnsembleAgent
"""
import numpy as np
from numpy.lib import utils

from tqdm import tqdm
import shutil

import torch
from torch import nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd

from .base import BaseAgent
# from models.multi_class_classification import MulticlassClassification
from models import *
from losses.cross_entropy import CrossEntropyLoss
from datasets.ensemble_data_loader import EnsembleLoader

from utils.utils import idx2class
from utils.utils import plot_loss_and_acc
from utils.avg_meter import AverageMeter


class MyEnsemble2Agent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.check_cuda()

        # Create an instance from the data loader
        self.data_loader = EnsembleLoader(self.config)
        self.data_loader.set_data_loader()

        NUM_FEATURES_A = len(config['train_col_A'])
        NUM_FEATURES_B = self.data_loader.X_train_B.shape[1]

        # if NUM_FEATURES_A + NUM_FEATURES_B != NUM_FEATURES:
        #     raise Exception("Feature sizes do not match")

        NUM_CLASSES = len(config['moods'])
        LEARNING_RATE = config['learning_rate']
        self.EPOCHS = self.config["epochs"]

        # Create an instance from the Model
        modelA_class = globals()[config["model_A"]]
        modelB_class = globals()[config["model_B"]]

        # Create models and load state_dicts
        modelA = modelA_class(NUM_FEATURES_A, NUM_CLASSES)
        # modelB = modelB_class(NUM_FEATURES, config["hidden_dim"], config["num_layer"], NUM_CLASSES)

        if config["model_B"] == 'MulticlassClassification':
            self.model = modelB_class(num_feature = NUM_FEATURES_B, num_class=NUM_CLASSES)
        elif config["model_B"] == 'LSTM':
            hidden_dim = config["hidden_dim"]
            num_layers = config["num_layer"]
            modelB = modelB_class(NUM_FEATURES_B, config["hidden_dim"], config["num_layer"], NUM_CLASSES)
            for i in range(len(list(modelB.parameters()))):
                print(list(modelB.parameters())[i].size())
            last_layer_size = 128
        elif config["model_B"] == 'RNN':
            hidden_dim = config["hidden_dim"]
            num_layers = config["num_layer"]
            modelB = modelB_class(num_features=NUM_FEATURES_B, hidden_dim=hidden_dim, n_layers=num_layers, num_class=NUM_CLASSES)
            last_layer_size = 192
        elif config["model_B"] == 'Transformer':
            nout = config["nout"]
            ninp = config["ninp"]
            nhead = config["nhead"]
            nhid = config["nhid"]
            nlayers = config["nlayers"]
            dropout = config["dropout"]
            modelB = modelB_class(nout, ninp, nhead, nhid, nlayers, dropout)

        # Load state dicts
        modelA = self.load_latest_model(modelA, config["checkpoint_dir"] + '/' + config["model_A_path"])
        modelB = self.load_latest_model(modelB, config["checkpoint_dir"] + '/' + config["model_B_path"])

        # Freeze these models
        for param in modelA.parameters():
            param.requires_grad_(False)

        for param in modelB.parameters():
            param.requires_grad_(False)

        ensemble_model = globals()[config["model"]]
        self.model = ensemble_model(modelA, modelB, last_layer_size, NUM_CLASSES)

        # self.modelA.to(self.device)
        # if self.device != 'cpu':
        #     self.modelA.cuda()

        # self.modelB.to(self.device)
        # if self.device != 'cpu':
        #     self.modelB.cuda()

        self.model.to(self.device)
        if self.device != 'cpu':
            self.model.cuda()

        # Create instance from the loss
        if config['imbalance']:
            self.criterion = nn.CrossEntropyLoss(weight=self.data_loader.class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Create instance from the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        # initialize my counters
        self.current_epoch = 0
        self.best_valid_acc = 0

        self.criterion.to(self.device)

        self.accuracy_stats = {
            'train': [],
            "val": []
        }
        self.loss_stats = {
            'train': [],
            "val": []
        }

        self.y_predict_test_list = []

        # Model Loading from the latest checkpoint if not found start from scratch.
        # self.load_checkpoint(self.config["checkpoint_file"])

        print(self.model)

    def check_cuda(self):
        # Check is cuda is available or not
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('********RUNNING ON:  {}'.format(self.device))

    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=0):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.config["checkpoint_dir"] + '/' + filename)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config["checkpoint_dir"] + '/' + filename,
                            self.config["checkpoint_dir"] + '/' + 'model_best.pth.tar')

    def load_checkpoint(self, filename):
        filename = self.config["checkpoint_dir"] + '/' + filename
        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            print("Checkpoint loaded successfully from '{}' at (epoch {}))\n"
                  .format(self.config["checkpoint_dir"] + '/', checkpoint['epoch']))
        except OSError as e:
            print("No checkpoint exists from '{}'. Skipping...".format(self.config["checkpoint_dir"] + '/'))
            print("**First time to train**")

    def load_latest_model(self, model, filename):
        # if self.device != 'cpu':
        map_location = self.device
        checkpoint = torch.load(filename, map_location=map_location)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def run(self):
        """
        This function will the operator
        :return:
        """
        try:
            if "mode" in self.config and self.config["mode"] == 'test':
                self.test()
            else:
                self.train_and_validate()
                self.test()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def train_and_validate(self):
        """
        Main training function, with per-epoch model saving
        """
        print("Begin Training")
        for e in tqdm(range(1, self.EPOCHS + 1)):
            self.current_epoch = e

            # Initialize stats
            train_epoch_loss = 0
            train_epoch_acc = 0

            # Set the model to be in training mode
            self.model.train()
            train_epoch_loss, train_epoch_acc = self.train_one_epoch(train_epoch_loss, train_epoch_acc)

            with torch.no_grad():
                # Initialize stats
                val_epoch_loss = 0
                val_epoch_acc = 0
                # set the model in training mode
                self.model.eval()
                val_epoch_loss, val_epoch_acc = self.validate(val_epoch_loss, val_epoch_acc)

            # Update stats
            self.loss_stats['train'].append(
                train_epoch_loss / (len(self.data_loader.train_loader_A)))
            self.loss_stats['val'].append(
                val_epoch_loss / (len(self.data_loader.val_loader_A)))
            self.accuracy_stats['train'].append(
                train_epoch_acc / (len(self.data_loader.train_loader_A)))
            self.accuracy_stats['val'].append(
                val_epoch_acc / (len(self.data_loader.val_loader_A)))

            print(f'Epoch {e + 0:03}: | \
                Train Loss: {train_epoch_loss / (len(self.data_loader.train_loader_A)):.5f} | \
                Val Loss: {val_epoch_loss / (len(self.data_loader.val_loader_A)):.5f} | \
                Train Acc: {train_epoch_acc / (len(self.data_loader.train_loader_A)):.3f}| \
                Val Acc: {val_epoch_acc / (len(self.data_loader.val_loader_A)):.3f}')

            is_best = val_epoch_acc > self.best_valid_acc
            if is_best:
                self.best_valid_acc = val_epoch_acc
            checkpoint_file = self.config["model"] + '_' + self.config["data_mode"] + '.pth.tar'
            self.save_checkpoint(checkpoint_file, is_best=is_best)

    def train_one_epoch(self, train_epoch_loss, train_epoch_acc):
        """
        One epoch training function
        """
        dataloader_iterator = iter(self.data_loader.train_loader_A)
        for i, dataB in enumerate(self.data_loader.train_loader_B):
            try:
                dataA = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(self.data_loader.train_loader_A)
                dataA = next(dataloader_iterator)

            X_train_batch_A, y_train_batch_A = dataA[0].to(self.device), dataA[1].to(self.device)
            X_train_batch_B, y_train_batch_B = dataB[0].to(self.device), dataB[1].to(self.device)
            self.optimizer.zero_grad()

            # model
            y_train_pred = self.model(X_train_batch_A, X_train_batch_B)

            # loss
            train_loss = self.criterion(y_train_pred, y_train_batch_A)
            if np.isnan(float(train_loss.item())):
                raise ValueError('Loss is nan during training...')

            # accuracy
            train_acc = self.multi_acc(y_train_pred, y_train_batch_A)

            # optimizer
            train_loss.backward()
            self.optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        # print("Training at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
        #     epoch_loss.val) + "- accuracy: " + str(epoch_acc.val))

        return train_epoch_loss, train_epoch_acc

    def validate(self, val_epoch_loss, val_epoch_acc):
        """
        One epoch validation
        :return:
        """
        dataloader_iterator = iter(self.data_loader.val_loader_A)
        for i, dataB in enumerate(self.data_loader.val_loader_B):
            try:
                dataA = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(self.data_loader.val_loader_A)
                dataA = next(dataloader_iterator)

            X_val_batch_A, y_val_batch_A = dataA[0].to(self.device), dataA[1].to(self.device)
            X_val_batch_B, y_val_batch_B = dataB[0].to(self.device), dataB[1].to(self.device)

            # model
            y_val_pred = self.model(X_val_batch_A, X_val_batch_B)

            # # loss
            val_loss = self.criterion(y_val_pred, y_val_batch_A)
            if np.isnan(float(val_loss.item())):
                raise ValueError('Loss is nan during validation...')

            # accuracy
            val_acc = self.multi_acc(y_val_pred, y_val_batch_A)

            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()

        # print("Validation results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
        #     epoch_loss.avg) + "- accuracy: " + str(epoch_acc.val))

        return val_epoch_loss, val_epoch_acc

    # function to calculate accuracy per epoch
    def multi_acc(self, y_pred, y_test):
        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

        correct_pred = (y_pred_tags == y_test).float()
        acc = correct_pred.sum() / len(correct_pred)

        acc = torch.round(acc * 100)

        return acc

    def test(self):
        with torch.no_grad():
            self.model.eval()
            dataloader_iterator = iter(self.data_loader.test_loader_A)
            for i, dataB in enumerate(self.data_loader.test_loader_B):
                try:
                    dataA = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(self.data_loader.test_loader_A)
                    dataA = next(dataloader_iterator)

                X_batch_A = dataA[0].to(self.device)
                X_batch_B = dataB[0].to(self.device)
                y_test_pred = self.model(X_batch_A, X_batch_B)
                _, y_pred_tags = torch.max(y_test_pred, dim=1)
                self.y_predict_test_list.append(y_pred_tags.cpu().numpy())
        self.y_predict_test_list = [a.squeeze().tolist() for a in self.y_predict_test_list]

    def save_confusion_matrix(self):
        cm = confusion_matrix(self.data_loader.y_test_A, self.y_predict_test_list)
        confusion_matrix_df = pd.DataFrame(cm).rename(columns=idx2class, index=idx2class)
        labels = confusion_matrix_df.columns.tolist()
        # plt.imshow(confusion_matrix_df)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title('Confusion matrix of the classifier')
        fig.colorbar(cax)
        # We want to show all ticks...
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Loop over data dimensions and create text annotations.
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, cm[i, j],
                               ha="center", va="center", color="w")

        # plt.show()
        fig.savefig('./figures/test_confusion_matrix_' + self.config['data_mode']   + '_' + self.config["model_A"]+ '_' + self.config["model_B"] + '.jpeg')
        fig.clear(True)

        print(confusion_matrix_df)

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        print("Please wait while finalizing the operation.. Thank you")
        plot_loss_and_acc(self.loss_stats, self.accuracy_stats, self.config["model"])
        checkpoint_file = self.config["checkpoint_file"]
        self.save_checkpoint(checkpoint_file)
        self.save_confusion_matrix()
        print("----------------------------------------")
        print(classification_report(self.data_loader.y_test_A, self.y_predict_test_list))
        self.data_loader.finalize()
