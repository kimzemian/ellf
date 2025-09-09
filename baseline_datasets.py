from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import os.path as osp
import torch
import numpy as np


def get_train_dataset_features(train_dataset, transform):
    features = np.concatenate([train_dataset.citations, train_dataset.accesses], axis=1)
    features = torch.from_numpy(transform(features))
    return features.float()


def get_train_dataset_features_github(train_dataset, transform):
    features = np.concatenate(
        [train_dataset.forks, train_dataset.stars, train_dataset.pushes], axis=1
    )
    features = torch.from_numpy(transform(features))
    return features.float()


def load_input_mean(horizon):
    input_mean = np.load(
        "/share/dean/arxiv-data/model_dev/baseline_benchmarking/inputs_mean.npy"
    )
    citations_mean = input_mean[:365]
    accesses_mean = input_mean[365:]  # TODO: why??
    citations_mean = citations_mean[:horizon]
    accesses_mean = accesses_mean[:horizon]
    combined = np.concatenate([citations_mean, accesses_mean])
    return combined


class TorchStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        """Compute mean and std from training data"""
        # self.mean = X.mean(dim=0).float()
        # self.std = X.std(dim=0).float()
        # self.std[self.std == 0] = 1.0  # Avoid division by zero
        self.mean = X.mean()
        self.std = X.std()

    def transform(self, X):
        """Standardize the data"""
        # import ipdb; ipdb.set_trace()
        mean, std = self.mean, self.std
        # if len(X.shape) == 2:
        #     mean = mean.unsqueeze(0)
        #     std = std.unsqueeze(0)
        return (X - mean) / (std + 1e-6)

    def inverse_transform(self, X_scaled):
        """Revert to original scale"""
        mean, std = self.mean, self.std
        # if len(X_scaled.shape) == 2:
        #     mean = mean.unsqueeze(0)
        #     std = std.unsqueeze(0)
        return X_scaled * (std + 1e-6) + mean

    # def fit_transform(self, X):
    #     self.fit(X)
    #     return self.transform(X)


class ArxivDataset(Dataset):
    def __init__(self, root, split, input_horizon):
        self.root = root
        self.split = split
        ds = load_from_disk(osp.join(root, split))
        self.citations = ds["citations_input"][:, :input_horizon]
        self.accesses = ds["accesses_input"][:, :input_horizon]
        self.labels = ds["citations_label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs = torch.log(
            1.0
            + torch.from_numpy(
                np.concatenate([self.citations[idx], self.accesses[idx]], axis=0)
            ).float()
        )
        orig_label = torch.tensor(self.labels[idx]).float()
        label = torch.log(1.0 + orig_label).unsqueeze(0)
        return inputs, label, orig_label


class GitHubDataset(Dataset):
    def __init__(self, root, split, input_horizon):
        self.root = root
        self.split = split
        ds = load_from_disk(osp.join(root, split))
        ds.set_format("numpy")
        self.forks = ds["forks_inputs"][:, :input_horizon]
        self.stars = ds["watches_inputs"][:, :input_horizon]
        self.pushes = ds["pushes_inputs"][:, :input_horizon]
        self.labels = ds["forks_labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs = torch.log(
            1.0
            + torch.from_numpy(
                np.concatenate(
                    [self.forks[idx], self.stars[idx], self.pushes[idx]], axis=0
                )
            ).float()
        )
        orig_label = torch.tensor(self.labels[idx]).float()
        label = torch.log(1.0 + orig_label).unsqueeze(0)
        return inputs, label, orig_label


class ArxivDatasetLinear(Dataset):
    def __init__(self, root, split, input_horizon, thresh=50, access_scale=100):
        self.root = root
        self.split = split
        self.thresh = thresh
        self.access_scale = access_scale
        self.labels_mean = 10.73
        ds = load_from_disk(osp.join(root, split))
        ds = ds.filter(lambda row: row["citations_label"] < thresh)
        self.citations = ds["citations_input"][:, :input_horizon]
        self.accesses = ds["accesses_input"][:, :input_horizon]
        self.labels = ds["citations_label"]
        self.input_mean = load_input_mean(input_horizon)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs = torch.from_numpy(
            np.concatenate(
                [self.citations[idx], self.accesses[idx] / self.access_scale], axis=0
            ).astype(np.float64)
            - self.input_mean
        ).float()
        orig_label = torch.tensor(self.labels[idx]).float()
        label = orig_label - self.labels_mean
        return inputs, label, orig_label


class ArxivDatasetAblation(Dataset):
    def __init__(self, root, split, input_horizon, ablation_name="accesses"):
        self.root = root
        self.split = split
        ds = load_from_disk(osp.join(root, split))
        self.ablation_name = ablation_name
        self.inputs = ds[f"{ablation_name}_input"][:, :input_horizon]
        self.labels = ds["citations_label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs = torch.log(1.0 + torch.from_numpy(self.inputs[idx]).float())
        orig_label = torch.tensor(self.labels[idx]).float()
        label = torch.log(1.0 + orig_label)
        return inputs, label, orig_label
