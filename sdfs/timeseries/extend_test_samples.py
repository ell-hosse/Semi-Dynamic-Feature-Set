from sdfs.timeseries.distances import find_closest_trend
import numpy as np
import torch

def concat_dfs_to_test_samples(Xw_test, Xw_train, dynamic_features):
    extended_Xw_test = []
    
    for i in range(Xw_test.shape[0]):
        test_sequence = Xw_test[i]
        closest_dynamic_features = find_closest_trend(Xw_train, test_sequence, dynamic_features_list=dynamic_features)
        concatenated_feature_set = np.concatenate((test_sequence, closest_dynamic_features.detach().numpy()), axis=1)
        extended_Xw_test.append(concatenated_feature_set)
        
    return torch.tensor(extended_Xw_test)