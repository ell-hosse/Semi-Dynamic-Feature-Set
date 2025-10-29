import numpy as np
import torch

def concat_feature_sets(static_feature_set, dynamic_feature_set):
    static_feature_set = np.array(static_feature_set, dtype=np.float32)
    dynamic_feature_set = np.array([item.detach().numpy() for item in dynamic_feature_set])
    concatenated_feature_set = np.concatenate((static_feature_set, dynamic_feature_set), axis=2)

    return torch.tensor(np.array(concatenated_feature_set))


