import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


import numpy

#global variables
test = None
train = None
mses = {}
features = None
target = "cnt"
bike_rentals = None
bike_rentals_file_name = "bike_rental_hour.csv"


def read_df(filename=bike_rentals_file_name):
    import os
    dir_name = os.path.dirname(os.path.abspath(__file__))
    full_file_name = os.path.join(dir_name, filename)
    df = pd.read_csv(full_file_name)
    return df


def explore_df(df):
    print("Few rows of the dataframe ...")
    print(df.head())
    print()
    print("correlation table for all columns ...")
    corr = bike_rentals.corr()
    print(corr)
    print("correlation table for target column '{}'...".format(target))
    print(corr[target].sort_values(ascending=False))
    plt.hist(bike_rentals[target])
    plt.show()


def assign_label(hour):
    if hour >= 0 and hour < 6:
        return 4
    elif hour >= 6 and hour < 12:
        return 1
    elif hour >= 12 and hour < 18:
        return 2
    elif hour >= 18 and hour <= 24:
        return 3


def get_train_test_df(df):
    # lets sample 80% of data for training
    train = df.sample(frac=.8)
    # remaining data, the data which is not part of train, becomes test data
    test = df.loc[~df.index.isin(train.index)]
    return train, test


def get_features_and_target(df):
    features = list(train.columns)
    # target is cnt column
    target = "cnt"
    # lets remove target column from list of features used to train the model
    features.remove(target)
    features.remove("casual")
    features.remove("registered")
    features.remove("dteday")
    return features, target


def model_lr(train, test, features, target):
    lr = LinearRegression()
    lr.fit(train[features], train[target])
    predictions = lr.predict(test[features])
    mse = numpy.mean((predictions-test[target])**2)
    return mse


def check_high_values(df, high_value):
    cnt_frequencey = df["cnt"].value_counts()
    high_rental = cnt_frequencey[cnt_frequencey >= high_value].count()
    high_rental_sum = cnt_frequencey[cnt_frequencey >= high_value].sum()
    low_rental = cnt_frequencey[cnt_frequencey < high_value].count()
    low_rental_sum = cnt_frequencey[cnt_frequencey < high_value].sum()
    total = cnt_frequencey.sum()
    high_to_low = high_rental/low_rental*100
    print("% of high renters to low renters (< {} hours) = {:0.2f}".format(
        high_value, high_to_low))
    print()
    print("""high renters who make up only {:0.2f} % have rented {:0.2f}% of total hours amounting to {:0.2f} hours. 
    Low renters who make up {:0.2f}  have rented {:0.2f} hours which is {:0.2f}% of total rentals""".format(
        high_to_low,
        high_rental_sum/total * 100,
        high_rental_sum,
        (100 - high_to_low),
        low_rental_sum,
        low_rental_sum/total * 100))


def model_dtr(train, test, features, target):
    dtr = DecisionTreeRegressor(min_samples_leaf=5)
    dtr.fit(train[features], train[target])
    predictions = dtr.predict(test[features])
    mse = numpy.mean((predictions - test[target]) ** 2)
    return mse


def model_rfr(train, test, features, target):
    rfr = RandomForestRegressor()
    rfr.fit(train[features], train[target])
    predictions = rfr.predict(test[features])
    mse = numpy.mean((predictions-test[target])**2)
    return mse


if __name__ == '__main__':
    # read data frame and explore datafram
    bike_rentals = read_df()
    explore_df(bike_rentals)

    # create additional column which provides better information about the bike rented hours
    bike_rentals["time_label"] = bike_rentals["hr"].apply(assign_label)

    # split data in train and test dataframes
    train, test = get_train_test_df(bike_rentals)
    # get feature and target columns
    features, target = get_features_and_target(bike_rentals)

    # start with LinearRegression model
    print()
    print("#"*60)
    mses["LinearRegression"] = model_lr(train, test, features, target)
    print("mse using LinearRegression = {:0.2f}".format(
        mses["LinearRegression"]))
    check_high_values(bike_rentals, 50)

    # move on to DecisionTresRegressior - This should reduce the error as it will account for
    # non linear columns
    print()
    print("#"*60)
    mses["DecisionTreeRegressor"] = model_dtr(train, test, features, target)
    print("mse using DecisionTreeRegressor = {:0.2f}".format(
        mses["DecisionTreeRegressor"]))

    # finally model on RandomForestRegressor - This should reduce the error as it will account for
    # non linear columns
    print()
    print("#"*60)
    mses["RandomForestRegressor"] = model_rfr(train, test, features, target)
    print("mse using RandomForestRegressor = {:0.2f}".format(
        mses["RandomForestRegressor"]))

    print()
    print("#"*60)
    print("Prediction summary... ")
    mses_value_series = pd.Series(mses, index=mses.keys()).sort_values()
    print(mses_value_series)
