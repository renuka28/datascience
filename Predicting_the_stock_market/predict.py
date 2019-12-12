from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
import pandas as pd
import os
import numpy as np
from datetime import datetime


def read_df(filename="AmesHousing.tsv"):
    dir_name = os.path.dirname(os.path.abspath(__file__))
    full_file_name = os.path.join(dir_name, filename)
    df = pd.read_csv(full_file_name)
    return df


def generate_indicators(df):

    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(by=["Date"], inplace=True)

    # lets use timeserie rolling method to compute mean for last five days
    day_5_mean = df["Close"].rolling(5).mean()
    day_5_mean = day_5_mean.shift(1)
    df["day_5_mean"] = day_5_mean

    # lets use timeserie rolling method to compute mean for last thirty days
    day_30_mean = df["Close"].rolling(30).mean()
    day_30_mean = day_30_mean.shift(1)
    df["day_30_mean"] = day_30_mean

    # lets use timeserie rolling method to compute mean for last 365 days
    day_365_mean = df["Close"].rolling(365).mean()
    day_365_mean = day_365_mean.shift(1)
    df["day_365_mean"] = day_365_mean

    # ratio between day_5/day_365
    df["mean_ratio_5_by_365"] = df["day_5_mean"]/df["day_365_mean"]

    # lets use timeserie rolling method to compute std for last five days
    day_5_std = df["Close"].rolling(5).std()
    day_5_std = day_5_std.shift(1)
    df["day_5_std"] = day_5_std

    # lets use timeserie rolling method to compute std for last thirty days
    day_30_std = df["Close"].rolling(30).std()
    day_30_std = day_30_std.shift(1)
    df["day_30_std"] = day_30_std

    # lets use timeserie rolling method to compute std for last 365 days
    day_365_std = df["Close"].rolling(365).std()
    day_365_std = day_365_std.shift(1)
    df["day_365_std"] = day_365_std

    # ratio between day_5_std/day_365_std
    df["std_ratio_5_by_365"] = df["day_5_std"]/df["day_365_std"]

    return df


def purge_df(df, start_date):
    clean = df[df["Date"] > start_date]
    return clean


def split_train_test(df, upto):
    train = df[df["Date"] < upto]
    test = df[df["Date"] >= upto]
    print(train.shape)
    print(train.head(2))
    print()
    print(test.shape)
    print(test.head(2))
    print()
    return train, test


def model_and_predict(train, test, feature_cols, target_col):
    # initialize model, train and test dataframes
    lr = linear_model.LinearRegression()
    train_features = train[feature_cols]
    train_target = train[target_col]

    test_features = test[feature_cols]
    test_target = test[target_col]

    # fit model
    lr.fit(train_features, train_target)
    predictions = lr.predict(test_features)

    # calculate mae
    mae = mean_absolute_error(test_target, predictions)
    return mae


def train_and_test(train, test):
    mae_values = {}
    # define features and target
    target_col = "Close"
    all_feature_columns = ['day_5_mean', 'day_30_mean', 'day_365_mean', 'mean_ratio_5_by_365',
                           'day_5_std', 'day_30_std', 'day_365_std', 'std_ratio_5_by_365']

    all_labels = ['5 day mean', '5 and 30 day mean', '5, 30 and 365 mean',
                  '5, 30, 365 mean + 5/365 mean ratio',
                  'all means + 5 day std', 'all means + 5, 30 day std',
                  'all means + 5, 30 and 365 std',
                  'all means + all std + 5/365 mean ratio + 5/365 std ratio',
                  ]

    # lets loop through feature columns and test it one by one, while adding features in each iteration
    for i in range(len(all_feature_columns)):
        feature_columns = all_feature_columns[0:i + 1]
        mae = model_and_predict(train, test, feature_columns, target_col)
        mae_values[all_labels[i]] = mae

    return mae_values


if __name__ == '__main__':

    df = read_df("sphist.csv")
    # Generate interesting indicators
    df = generate_indicators(df)

    # remove rows which have NaN in indicators
    start_date = datetime(year=1951, month=6, day=18)
    df = purge_df(df, start_date)

    # split df into train and test dataframes
    split_date = datetime(year=2013, month=1, day=1)
    train, test = split_train_test(df, split_date)

    # ready to train and test now
    mae_values = train_and_test(train, test)

    # format return values as pd.Series and display
    mae_value_series = pd.Series(mae_values, index=mae_values.keys())
    print(mae_value_series)
