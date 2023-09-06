import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST,CIFAR10
from torch.utils.data import DataLoader, random_split
import random
import copy

def setup_seed(seed):
    # Set random seeds for different packages to ensure reproducibility
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     random.seed(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.enabled=True

class LoadData:
    # Download, split and shuffle dataset into initial training, validation, test and pool

    def __init__(self, dataset: str = 'MNIST',val_size: int = 1000):
        self.val_size = val_size 
        self.dataset = dataset       
        self.train, self.test = self.download_dataset() 
         
        (
            self.X_init,
            self.y_init,
            self.X_val,
            self.y_val,
            self.X_pool,
            self.y_pool,
            self.X_test,
            self.y_test,
        ) = self.split_and_load_dataset()
        

    def tensor_to_np(self, tensor_data: torch.Tensor):
        # Transform tensor to numpy array to aviod data type error

        np_data = tensor_data.detach().numpy()
        return np_data


    def check_folder(self) -> bool:
        # Check whether dataset folder exists, skip download if existed

        if self.dataset=='MNIST':
            if os.path.exists("MNIST/"):
                return False
            return True

        if self.dataset=='CIFAR10':
            if os.path.exists("CIFAR10/"):
                return False
            return True


    def download_dataset(self):
        # Download and normalize data

        download = self.check_folder()
        if self.dataset=='MNIST':
            transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            train=MNIST(".", train=True, download=download, transform=transform)
            test=MNIST(".", train=False, download=download, transform=transform)
        else:
            transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
            train = CIFAR10(".", train=True, download=download, transform=transform)
            test = CIFAR10(".", train=False, download=download, transform=transform)
        return train, test

    def split_and_load_dataset(self):
        # Randomly sample data points from the training set to create intial training, validation and pool set

        size = 2 if self.dataset == 'MNIST' else 10
        all_size = 60000 if self.dataset == 'MNIST' else 50000
        idx_all=np.arange(all_size)

        initial_idx = np.array([], dtype=np.int)

        for i in range(10):
            idx = np.random.choice(
                np.where(np.array(self.train.targets) == i)[0], size=size, replace=False
            )
            initial_idx = np.concatenate((initial_idx, idx))

        intial_data=copy.deepcopy(self.train)

        intial_data.data=intial_data.data[initial_idx]
        intial_data.targets=torch.tensor(intial_data.targets)[initial_idx]

        # Initial training set
        init_loader = DataLoader(dataset=intial_data, batch_size=10*size, shuffle=True)
        X_init, y_init = next(iter(init_loader))

        print(f"Initial training data points: {X_init.shape[0]}")
        print(f"Data distribution for each class: {np.bincount(y_init)}")

        idx_left=np.delete(idx_all,initial_idx)
        self.train.data=self.train.data[idx_left]

        # The targets of CIFAR-10 is a list, so trandform it
        self.train.targets=torch.tensor(self.train.targets)[idx_left]

        # Validation and pool set
        val_set, pool_set = random_split(
            self.train, [self.val_size, all_size-size*10-self.val_size]
        )

        val_loader = DataLoader(dataset=val_set, batch_size=self.val_size, shuffle=True)
        pool_loader = DataLoader(
            dataset=pool_set, batch_size=all_size-size*10-self.val_size, shuffle=True
        )

        # Test set
        test_loader = DataLoader(
          dataset=self.test, batch_size=10000, shuffle=True
        )

        X_val, y_val = next(iter(val_loader))
        X_pool, y_pool = next(iter(pool_loader))
        X_test, y_test = next(iter(test_loader))
        return X_init, y_init, X_val, y_val, X_pool, y_pool, X_test, y_test


    def load_all(self):
        # Load all data

        return (
            self.tensor_to_np(self.X_init),
            self.tensor_to_np(self.y_init),
            self.tensor_to_np(self.X_val),
            self.tensor_to_np(self.y_val),
            self.tensor_to_np(self.X_pool),
            self.tensor_to_np(self.y_pool),
            self.tensor_to_np(self.X_test),
            self.tensor_to_np(self.y_test),
        )






