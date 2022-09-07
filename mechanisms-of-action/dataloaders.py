import numpy as np
from path import Path
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from data_helpers import train_val_split, remove_id_and_fold
import torch


class DrugsDataset(Dataset):
    def __init__(self, features, targets=None):
        super(DrugsDataset, self).__init__()
        if targets is not None:
            assert features.shape[0] == targets.shape[0]
            
        self.features = remove_id_and_fold(features)   
        self.targets = targets if targets is None else remove_id_and_fold(targets)  
          
            
    def __len__(self):
        return self.features.shape[0]
    
    
    def __getitem__(self, idx):
        tabular_features = torch.tensor(self.features.iloc[idx,:].values.astype(np.float32) )       
        if self.targets is not None:
            target = torch.tensor(self.targets.iloc[idx, :].values.astype(np.float32))
            return {"tabular_features" : tabular_features, "target" : target}
        else:
            return {"tabular_features" : tabular_features}



def get_trainloaders(df, targets, val_fold, batch_size):
    train_df, val_df, train_targets, val_targets = train_val_split(df, targets, val_fold)
    
    train_dataset = DrugsDataset(train_df, train_targets)
    val_dataset = DrugsDataset(val_df, val_targets)
    
    train_loader = DataLoader(train_dataset, sampler=SequentialSampler(train_dataset), batch_size=batch_size)
    val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
    
    return train_loader, val_loader


def get_testloader(test_df, batch_size):
    test_dataset = DrugsDataset(test_df)
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)
    
    return test_loader