import torch
from torch.utils.data import Dataset
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MyDataset(Dataset):
    def __init__(self, Xs, ys, split, T=50):
        self.Xs = Xs
        self.ys = ys
        print("X shape:", self.Xs.shape)
        print("y shape", self.Xs.shape)        

        test_size = 0.2
        rand_seed = 42
        X_train, X_test, y_train, y_test = train_test_split(self.Xs, self.ys, test_size=test_size, random_state=rand_seed, shuffle=True)
        
        if split == 'train':
            print('loading train data...')
            self.x = torch.from_numpy(X_train).float()
            self.y = torch.from_numpy(y_train).float()
        elif split == 'test':
            print('loading test data...')
            self.x = torch.from_numpy(X_test).float()
            self.y = torch.from_numpy(y_test).float()
        
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]


