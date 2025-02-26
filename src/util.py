import os

import sklearn.metrics
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd

def create_directory(directory_new):
    current_path = os.getcwd()
    try:
        os.mkdir(current_path + "/" + directory_new)
    except OSError:
        print("Creation of the directory %s failed " % current_path + directory_new)
    else:
        print("Successfully created the directory %s " % current_path + directory_new)

def import_data_pd(filename):
    data = pd.read_csv(filename, index_col=0)
    return data

def one_hot_max(output):
    values, indices = output.max(0)
    one_hot_pred = torch.zeros(output.shape)
    one_hot_pred[indices] = 1
    return values * one_hot_pred

def confusion_matrix(labels, predictions, label_order_encountered, label_order, path):
    fig, ax = plt.subplots(figsize=(20, 15))

    na_lines = [label_order.index(l) for l in label_order if l not in label_order_encountered]
    na_lines = sorted(na_lines, reverse=True)
    cm = sklearn.metrics.confusion_matrix(y_true=labels, y_pred=predictions)
    cmn = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100

    cm = cm.tolist()
    cmn = cmn.tolist()
    for na_line in na_lines:
        cm.insert(na_line, [0] * len(cm[0]))
        cmn.insert(na_line, [0] * len(cm[0]))
        for line in cm:
            line.insert(na_line, 0)
        for line in cmn:
            line.insert(na_line, 0)
    cmn = np.array(cmn)

    annot_perc = []
    for row in cmn.tolist():
        annot_perc.append([])
        for e in row:
            if e == 0:
                annot_perc[-1].append(f"")
            elif e % 1 == 0:
                annot_perc[-1].append(f"{int(e)}%")
            else:
                annot_perc[-1].append(f"{int(round(e, 0))}%") # round(e, 1)

    sn.heatmap(cmn, annot=annot_perc, annot_kws={'va': 'center', "size": 14, "fontweight": "bold"}, cbar=False, fmt="", cmap='Oranges', vmin=0, vmax=100)
    sn.heatmap(cmn, annot=False, annot_kws={"size": 20}, cmap='Oranges', fmt='.2f', xticklabels=label_order, yticklabels=label_order, vmin=0, vmax=100)  # fmt https://stackoverflow.com/questions/29647749/seaborn-showing-scientific-notation-in-heatmap-for-3-digit-numbers

    len_gid = len(label_order)
    plt.hlines(y=np.arange(0, len_gid + 1) + 0.5, xmin=np.full(len_gid + 1, 0) - 0.5, xmax=np.full(len_gid + 1, len_gid + 1) - 0.5, color=[0, 0, 0, 0.1], linestyles="dashed")
    plt.vlines(x=np.arange(0, len_gid + 1) + 0.5, ymin=np.full(len_gid + 1, 0) - 0.5, ymax=np.full(len_gid + 1, len_gid + 1) - 0.5, color=[0, 0, 0, 0.1], linestyles="dashed")

    plt.hlines(y=np.arange(0, len_gid + 1), xmin=np.full(len_gid + 1, 0), xmax=np.full(len_gid + 1, len_gid), color=[0, 0, 0, 0.3])
    plt.vlines(x=np.arange(0, len_gid + 1), ymin=np.full(len_gid + 1, 0), ymax=np.full(len_gid + 1, len_gid), color=[0, 0, 0, 0.3])

    ax.tick_params(axis='both', which='major', labelsize=16)
    fig.tight_layout()

    plt.savefig(f"{path}/confusion_matrix.png")

def hist_conficence(data, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.hist(data, range=(0, 1.001), bins=100, rwidth=0.9)
    plt.xticks([0.1 * i for i in range(0, 11)])
    plt.margins(x=0)
    plt.ylabel("Number of Predictions")
    plt.xlabel("Confidence of Prediction")
    plt.savefig(path)
    plt.close()
