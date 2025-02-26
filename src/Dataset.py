from torch.utils.data import Dataset
import pandas as pd
import torch

class FeatureDataset(Dataset):

    def __init__(self, filename):
        file_input = pd.read_csv(filename)
        file_input['X'] = pd.factorize(file_input['X'])[0]
        x = file_input.iloc[0:2595, 1:3048].values
        y = file_input.iloc[0:2595, 0].values

        x_train = x
        y_train = y

        self.x_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

class Dataset(Dataset):
    def __init__(self, X=None, Y=None, one_hot=False):
        self.X = X
        self.features = list(X.index)
        self.length_ = len(X.iloc[0].tolist())
        self.label_name = sorted(list(set(Y.iloc[0, :].tolist())))
        if one_hot:
            prefix_tmp = "prefix_that_needs_to_be_removed_afterwards_and_should_not_match_the_label"
            y_tmp = pd.get_dummies(Y.T, prefix=prefix_tmp, prefix_sep="").T
            y_tmp.index = [i.replace(prefix_tmp, "") for i in list(y_tmp.index)]
            self.Y = y_tmp.replace({True: 1, False: 0})
        else:
            self.Y = Y

    def __len__(self):
        # Denotes the total number of samples
        return self.length_

    def __getitem__(self, index): # vllt mit get index + label
        # Generates one sample of data
        # Select sample
        if isinstance(index, int):
            index = [index]

        if isinstance(index, slice):
            index = list(range(index.stop)[index])

        x = torch.Tensor(self.X.iloc[:, index].T.values)
        y = torch.Tensor(self.Y.iloc[:, index].T.values)

        return x, y