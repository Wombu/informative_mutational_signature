import numpy as np
import torch
import pandas as pd
from src.model import NN1
from src.Dataset import FeatureDataset
from src import util

from zennit.composites import EpsilonPlus
from zennit.attribution import Gradient

label_order = ["Ovary-AdenoCA", "CNS-PiloAstro", "Liver-HCC", "Panc-Endocrine", "Kidney-RCC", "Prost-AdenoCA", "Thy-AdenoCA", "ColoRect-AdenoCA",
                "Lymph-BNHL", "Uterus-AdenoCA", "Breast-AdenoCA", "Lung-AdenoCA", "Panc-AdenoCA", "Eso-AdenoCA", "Head-SCC", "CNS-Medullo",
               "CNS-GBM", "Skin-Melanoma", "Lymph-CLL", "Kidney-ChRCC", "Stomach-AdenoCA", "Lung-SCC", "Bone-Osteosarc", "Myeloid-MPN"]

pred_confidence = []  # for confidence histogram
pred_confidence_threshold = 0.9 # confidence threshold for evaluation
quantitative_lrp_threshold = 0.9  # value that should be explained for the quantitative-LRP

data_name = "dataset_WGS_MS+Bins"
path_data = f"WGS_MS+Bins/"

file_input = pd.read_csv(f"data/{path_data}/{data_name}.csv")
feature_names = list(file_input.columns[1:])

output_name = f"{data_name}_{pred_confidence_threshold}"

path_output = f"output/{output_name}"
util.create_directory(path_output)

softmax = torch.nn.Softmax(dim=0)
relevances = {l: [] for l in label_order}
relevances_pred = {l: [] for l in label_order}

predictions_total = []
labels_total = []
for i in range(1, 11, 1):
    ann = torch.load(f"data/{path_data}/ann{i}.pt")  # import individual model
    validation_loader = torch.load(f"data/{path_data}/val_loader{i}.pth") # import validation dataset
    train_loader = torch.load(f"data/{path_data}/train_loader{i}.pth") # import training dataset

    dataset = validation_loader.dataset
    indizes = validation_loader.batch_sampler.sampler.indices
    dataset, label = dataset[indizes]
    label = list(label.numpy())

    predictions = []  # predictions
    labels = []  # true label

    for d, data_tmp in enumerate(dataset):
        composite = EpsilonPlus()
        with Gradient(model=ann, composite=composite, attr_output=util.one_hot_max) as attributor:
            out, relevance = attributor(input=data_tmp)
            pred_softmax = softmax(out).detach().numpy()
            pred_max = pred_softmax.max()
            pred_confidence.append(pred_max)
            if pred_max < pred_confidence_threshold:
                continue
            else:
                predictions.append(pred_softmax.argmax())
                labels.append(label[d])

            pred = out.detach().numpy()
            relevance_tmp = relevance.detach().numpy()
            sum_abs_relevance_tmp = np.sum(np.abs(relevance_tmp))
            relevance_tmp = relevance_tmp / sum_abs_relevance_tmp # lrp normalisation

            relevances[label_order[pred.argmax()]].append(relevance_tmp)
            relevances_pred[label_order[pred.argmax()]].append(pred[pred.argmax()] / sum_abs_relevance_tmp)

    predictions_total = predictions_total + predictions
    labels_total = labels_total + labels

util.hist_conficence(data=pred_confidence, path=f"{path_output}/confidence.png")

labels_encountered = set(labels_total + predictions_total) # selection needed for matplotlib since you cant create empty rows easily
label_order_encountered = [l for l in label_order if label_order.index(l) in labels_encountered]
util.confusion_matrix(labels=labels_total, predictions=predictions_total, label_order_encountered=label_order_encountered, label_order=label_order, path=path_output)

for pred_label in relevances.keys():
    df_dict = {}
    df = pd.DataFrame(relevances[pred_label], columns=feature_names)

    df_dict["median"] = df.median(axis=0)
    df_dict["mean"] = df.mean(axis=0)
    df_dict["std"] = df.std(axis=0)
    df_dict["min"] = df.min(axis=0)
    df_dict["max"] = df.max(axis=0)
    df_new = pd.DataFrame.from_dict(df_dict)

    df_new.to_csv(f"{path_output}/relevance_{pred_label}.csv")

prediction_feature_amount = {}
quantitative_lrp = {}
for pred_label in relevances.keys():
    df = pd.DataFrame(relevances[pred_label], columns=feature_names)  # df of all predictions for one label
    df_abs = df.abs()
    shape = df_abs.shape
    columns_fetures = list(df.columns)
    df_counts_pos = pd.DataFrame(0, index=np.arange(shape[0]), columns=df.columns)
    for i in range(shape[0]):
        row_values = df.iloc[i, :]
        row = df_abs.iloc[i, :]
        row_ranked = row.rank(axis=0, method="min", ascending=False)
        row_ranked_sorted = row_ranked.sort_values(ascending=True) # ranking of the most influencial relevances
        row_ranked_max = row_ranked.max()
        importance_limit = relevances_pred[pred_label][i] * quantitative_lrp_threshold
        importance_limit_tmp = 0
        for k in range(1, int(row_ranked_max)):
            if importance_limit_tmp <= importance_limit:  # threshold testing
                feature_tmp = row_ranked[row_ranked == k].index
                feature_tmp = list(feature_tmp)
                for f_tmp in feature_tmp: # needed if two or more features have the same rank
                    value_abs = row_values[f_tmp]
                    feature_index_tmp = columns_fetures.index(f_tmp)
                    importance_limit_tmp = importance_limit_tmp + value_abs

                    value_tmp = df.iloc[i, feature_index_tmp]
                    if value_tmp > 0:
                        df_counts_pos.iloc[i, feature_index_tmp] += 1
            else:
                break  # break if threshold is reached

    df_zeroes_sum = df_counts_pos.sum(axis=0)/shape[0]
    quantitative_lrp[pred_label] = df_zeroes_sum

    print(f"{pred_label} done")

columns = list(relevances.keys())
quantitative_lrp = pd.DataFrame.from_dict(quantitative_lrp, orient="columns")
quantitative_lrp.to_csv(f"{path_output}/quantitative_lrp.csv")
