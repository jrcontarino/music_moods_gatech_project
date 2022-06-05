import pandas as pd
import matplotlib.pyplot as plt

class2idx = {
    'calm':0,
    'energetic':1,
    'happy':2,
    'sad':3
}
# class2idx = {}

idx2class = {v: k for k, v in class2idx.items()}

def set_class2idx(list_of_moods):
    init = 0
    for mood in list_of_moods:
        class2idx[mood] = init
        init += 1
    print(class2idx)


def plot_loss_and_acc(loss_stats, accuracy_stats, name=''):
        # Create dataframes
        train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        # Plot the dataframes
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
        axes[0].plot(train_val_acc_df[train_val_acc_df["variable"]=='train']["epochs"], train_val_acc_df[train_val_acc_df["variable"]=='train']["value"],label = "train")
        axes[0].plot(train_val_acc_df[train_val_acc_df["variable"]=='val']["epochs"], train_val_acc_df[train_val_acc_df["variable"]=='val']["value"],label = "val")
        axes[0].set_title('Train-Val Accuracy/Epoch')
        axes[0].legend()
        # axes[0].set_ylim([0, 100])


        axes[1].plot(train_val_loss_df[train_val_loss_df["variable"]=='train']["epochs"], train_val_loss_df[train_val_loss_df["variable"]=='train']["value"],label = "train")
        axes[1].plot(train_val_loss_df[train_val_loss_df["variable"]=='val']["epochs"], train_val_loss_df[train_val_loss_df["variable"]=='val']["value"],label = "val")
        axes[1].legend()
        axes[1].set_title('Train-Val Loss/Epoch')
        # axes[1].set_ylim([0, 1])

        fig.savefig('./figures/train_loss_and_acc_'+name+'.jpeg')
        fig.clear(True)

