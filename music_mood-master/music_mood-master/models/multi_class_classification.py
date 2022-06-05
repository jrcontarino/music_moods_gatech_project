import torch
import torch.nn as nn


class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()
        # # model 1
        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

        # # model 2
        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=3, padding=1)
        # self.batch1 = nn.BatchNorm1d(40)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv1d(in_channels=40, out_channels=80, kernel_size=3, padding=1)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(p=0.2)
        # self.avg1 = nn.MaxPool1d(kernel_size=2, stride=2)
        #
        # self.conv3 = nn.Conv1d(in_channels=80, out_channels=160, kernel_size=3, padding=1)
        # self.batch3 = nn.BatchNorm1d(160)
        # self.relu3 = nn.ReLU(inplace=True)
        # self.conv4 = nn.Conv1d(in_channels=160, out_channels=160, kernel_size=3, padding=1)
        # self.relu4 = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(p=0.2)
        # self.avg2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.L = nn.Linear(in_features=320, out_features=num_class)

        # # model 3
        # self.layer_1 = nn.Linear(num_feature, 1024)
        # self.tanh1 = nn.Hardtanh(min_val=-0.5, max_val=0.5)
        # self.dropout1 = nn.Dropout(p=0.2)
        #
        # self.layer_2 = nn.Linear(1024, 128)
        # self.tanh2 = nn.Hardtanh(min_val=-0.5, max_val=0.5)
        # self.dropout2 = nn.Dropout(p=0.2)
        #
        # self.layer_out = nn.Linear(128, num_class)


    def forward(self, x):
        # # model 1

        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        # # model 2

        # x = x.unsqueeze(dim=1)
        # x = self.conv1(x)
        # x = self.batch1(x)
        # x = self.relu1(x)
        # x = self.conv2(x)
        # x = self.relu2(x)
        # x = self.dropout(x)
        # x = self.avg1(x)
        #
        # x = self.conv3(x)
        # x = self.batch3(x)
        # x = self.relu3(x)
        # x = self.conv4(x)
        # x = self.relu4(x)
        # x = self.dropout(x)
        # x = self.avg2(x)
        #
        # x = torch.flatten(x, start_dim=1)
        # x = self.L(x)

        # # model 3

        # x = self.layer_1(x)
        # x = self.tanh1(x)
        # x = self.dropout1(x)
        #
        # x = self.layer_2(x)
        # x = self.tanh2(x)
        # x = self.dropout2(x)
        #
        # x = self.layer_out(x)

        return x
