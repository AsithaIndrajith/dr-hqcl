import os
import urllib.request
import zipfile
import torch
import medmnist
import numpy as np
import torch as nn
import pandas as pd
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from medmnist import INFO
from sklearn.preprocessing import StandardScaler

def load_dataset( input_shape, batch_size, data_dir, data_transforms ):
    image_datasets = {
    x if x == "train" else "val": datasets.ImageFolder(
        os.path.join(data_dir, x), data_transforms[x]
    )
    for x in ["train", "val"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes
    y_train = image_datasets["train"].targets
    y_test = image_datasets["val"].targets

    # Initialize dataloader
    dataloaders = {
    x: nn.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
    for x in ["train", "val"]
    }

    train_dataset = image_datasets["train"]
    test_dataset = image_datasets["val"]

    train_loader = dataloaders['train']
    test_loader = dataloaders['val']

    print("Training dataset size: " + str(len(train_dataset)))
    print("Testing datset size: " + str(len(test_dataset)))

    return train_dataset, test_dataset, train_loader, test_loader, dataset_sizes, class_names, y_train, y_test

def get_stats(train_loader, input_shape):
    # https://kozodoi.me/blog/20210308/compute-image-stats
    psum    = nn.tensor([0.0, 0.0, 0.0])
    psum_sq = nn.tensor([0.0, 0.0, 0.0])

    # loop through images
    for inputs, labels in train_loader:
        psum    += inputs.sum(axis        = [0, 2, 3])
        psum_sq += (inputs ** 2).sum(axis = [0, 2, 3])

    ####### FINAL CALCULATIONS

    # pixel count
    count = len(train_loader.dataset) * input_shape[0] * input_shape[0]

    # mean and std
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = nn.sqrt(total_var)

    # output
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))

    return total_mean, total_std


def download_and_extract_data(url, destination_folder):
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # File path to save the downloaded zip file
    zip_file_path = os.path.join(destination_folder, os.path.basename(url))

    try:
        # Download the zip file
        urllib.request.urlretrieve(url, zip_file_path)

        # Extract the contents of the zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(destination_folder)

        # Remove the downloaded zip file
        os.remove(zip_file_path)

        print(f"Data downloaded and extracted to '{destination_folder}' folder.")
    except Exception as e:
        print(f"Error: {e}")

class SmallDataset(Dataset):
    def __init__(self, samples_per_class, dataset):
        self.samples_per_class = samples_per_class
        self.dataset = dataset
        self.class_indices = self._get_class_indices()

    def _get_class_indices(self):
        class_indices = {}
        for idx, (_, label_arr) in enumerate(self.dataset):
            label = label_arr[0]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices

    def __len__(self):
        return len(self.class_indices) * self.samples_per_class

    def __getitem__(self, idx):
        class_idx = idx // self.samples_per_class
        label = list(self.class_indices.keys())[class_idx]
        indices = self.class_indices[label]
        selected_idx = indices[idx % self.samples_per_class]
        return self.dataset[selected_idx]

def create_small_dataset( samples_per_class, dataset, batch_size ):
    # Get the indices of samples for each class
    class_indices = {}
    for idx, (_, label_arr) in enumerate(dataset):
        label = label_arr[0]
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    # Select 20 samples from each class
    selected_indices = []
    for label, indices in class_indices.items():
        selected_indices.extend(indices[:samples_per_class])

    # Create a SubsetRandomSampler from the selected indices
    sampler = SubsetRandomSampler(selected_indices)

    # Create a new data loader with the custom sampler
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

def get_data_class(data_flag):
    # data_flag = 'breastmnist'
    download = True

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])
    return DataClass

def convert_2048_features(train_loader, device, encoder):
    X_2048_train = []
    y_2048_train = []
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            X_2048_train.extend(encoder(inputs))
            y_2048_train.extend(labels.cpu().numpy())

    x_2048_train = np.vstack([tensor.cpu().numpy() for tensor in X_2048_train])
    y_2048_train = np.reshape(y_2048_train, (len(y_2048_train), 1))
    return x_2048_train, y_2048_train

def scale_dataset(dataset):
    scaler = StandardScaler()
    scaled_dataset = scaler.fit_transform( dataset )
    return scaled_dataset

def create_new_dataframe_and_save_csv(X, y, num_cols, models_dir, file_name):
    merged_np_train_1 = np.concatenate((X, y), axis=1)

    merged_df_1 = pd.DataFrame(merged_np_train_1, columns =['f'+str(i) for i in range(num_cols)]+['y'])
    merged_df_1.to_csv(f'{models_dir}/{file_name}.csv')
    return merged_df_1