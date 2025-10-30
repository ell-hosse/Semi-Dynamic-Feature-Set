from sdfs.timeseries.models import SDFS
from sdfs.timeseries.train import train
from sdfs.timeseries.dynamic_features import init_randomly
from sdfs.timeseries.concatenate_feature_sets import concat_feature_sets
from sdfs.timeseries.extend_test_samples import concat_dfs_to_test_samples


def sdfs(Xw_train, Xw_val, Xw_test, y_train, y_val, y_test,
         dynamic_input_size=5):
    """
    Semi-Dynamic Feature Set (SDFS) for time-series regression using an LSTM-based model.
    """

    dynamic_features = init_randomly(Xw_train, dynamic_input_size=dynamic_input_size)
    print("Dynamic features have been initialized successfully.")

    model = SDFS(
        static_input_size=Xw_train.shape[2],
        dynamic_input_size=dynamic_input_size,
        hidden_size=64,
        num_layers=1,
        dropout=0.1,
        bidirectional=False
    )

    final_loss, loss_values, dynamic_features_trained = train(
        model, Xw_train, y_train, Xw_val, y_val, dynamic_features
    )

    extended_Xw_train = concat_feature_sets(Xw_train, dynamic_features_trained)
    extended_Xw_val = concat_dfs_to_test_samples(Xw_val, Xw_train, dynamic_features_trained)
    extended_Xw_test = concat_dfs_to_test_samples(Xw_test, Xw_train, dynamic_features_trained)

    print("Feature expansion has been completed successfully.")

    return extended_Xw_train, extended_Xw_val, extended_Xw_test

