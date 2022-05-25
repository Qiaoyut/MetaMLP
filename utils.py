import numpy as np
import torch


def split_dataset(data, setting):
    num_node = data.y.size(0)
    train_set = torch.where(data.train_mask)[0]
    val_set = torch.where(data.val_mask)[0]
    test_set = torch.where(data.test_mask)[0]
    if setting == 'super':
        curr_all = set(torch.cat([train_set, val_set, test_set], dim=0).data.cpu().numpy())
        all_set = set(np.arange(num_node))
        train_set_ = list(set(all_set).difference(curr_all))
        train_set = train_set_ + train_set.data.cpu().numpy().tolist()
        train_set = np.array(train_set)
        train_set = torch.from_numpy(train_set)

    return train_set, val_set, test_set
