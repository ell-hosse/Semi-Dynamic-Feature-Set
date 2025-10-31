from examples.timeseries.AirQualityUCI import *
from sdfs.timeseries.feature_expansion import sdfs


def run_example(dynamic_input_size):
    set_seed(SEED)
    #XLSX_PATH = "examples/timeseries/AirQualityUCI.xlsx"

    print("Loading AirQualityUCI (full, cleaned)...")
    df = load_airquality(XLSX_PATH)

    train_df, val_df, test_df = split_time_80_10_10(df)

    scaler = StandardScaler().fit(train_df[FEATURES].values)

    def build(split_df: pd.DataFrame):
        X = scaler.transform(split_df[FEATURES].values)
        y = split_df[TARGET_COL].values
        return make_windows(X, y, SEQ_LEN, HORIZON)

    Xw_train, yw_train = build(train_df)
    Xw_val, yw_val = build(val_df)
    Xw_test, yw_test = build(test_df)

    expanded_Xw_train, expanded_Xw_val, expanded_Xw_test = sdfs(Xw_train, Xw_val, Xw_test,
                                                                yw_train, yw_val, yw_test,
                                                                dynamic_input_size=dynamic_input_size)

    main(expanded_Xw_train.numpy(), expanded_Xw_val.numpy(), expanded_Xw_test.numpy(),
         yw_train, yw_val, yw_test, input_size=len(FEATURES)+dynamic_input_size)


if __name__ == '__main__':
    run_example(dynamic_input_size=2)