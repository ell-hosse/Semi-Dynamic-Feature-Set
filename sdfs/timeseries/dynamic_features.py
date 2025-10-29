import torch

def init_randomly(Xw_train, dynamic_input_size):
    dynamic_features_list = [torch.randn((len(Xw_train[0]), dynamic_input_size),
                                         requires_grad=True) for _ in range(len(Xw_train))]

    return dynamic_features_list
