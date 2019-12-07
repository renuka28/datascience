import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import os


def read_data(filename, cols):
    dir_name = os.path.dirname(os.path.abspath(__file__))
    full_file_name = os.path.join(dir_name, filename)
    df = pd.read_csv(full_file_name, names=cols)
    # print(df.info())
    return df


def get_car_df():
    pd.options.display.max_columns = 99
    cols = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
            'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
            'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
    cars = read_data('imports-85.data', cols=cols)
    # print(cars.head(5))

    continuous_values_cols = ['normalized-losses', 'wheel-base', 'length', 'width',
                              'height', 'curb-weight', 'bore', 'stroke',
                              'compression-rate', 'horsepower', 'peak-rpm',
                              'city-mpg', 'highway-mpg', 'price']
    numeric_cars = cars[continuous_values_cols]
    return numeric_cars


def clean_df(df):
    df = df.replace('?', np.nan)
    df = df.astype('float')
    df = df.dropna(subset=['price'])
    # print(df.isnull().sum())
    df = df.fillna(df.mean())
    # print(df.isnull().sum())
    # print(numeric_cars.isnull().sum())
    # normalize all columns excep price
    price_col = df['price']
    df = (df - df.min())/(df.max() - df.min())
    df['price'] = price_col
    return df


def divide_train_test(df, divide_by=2):
     # lets shuffel the dataset so that we can get random set of rows in test
    # training datsets
    index_permutation = np.random.permutation(df.index)
    randomized_df = df.reindex(index_permutation)

    # lets divide the df into two equal part, one for training and another for
    # testing
    upto = int(len(randomized_df)/divide_by)
    training_df = randomized_df[:upto]
    test_df = randomized_df[upto:]
    return training_df, test_df


def knn_train_test(df, training_df, training_col, test_df, target_col, k=5):
    knn_model = KNeighborsRegressor(n_neighbors=k)
    np.random.seed(1)

    if isinstance(training_col, str):
        # quickly get a list out of training_col
        # we know the columns cannot contain some random chars :)
        # but split will quickly convert string to list
        training_col = training_col.split(",!,!,!,!,&.1!")

    # lets fit KNeigborsRegressor model and predict
    knn_model.fit(training_df[training_col], training_df[target_col])
    prediction = knn_model.predict(test_df[training_col])

    # lets calculate RMSE
    mse = mean_squared_error(test_df[target_col], prediction)
    rmse = np.sqrt(mse)
    return rmse


def get_feature_avg_rmse(k_rmse_results):
    feature_avg_rmse = {}
    for k, v in k_rmse_results.items():
        avg_rmse = np.mean(list(v.values()))
        feature_avg_rmse[k] = avg_rmse
    return feature_avg_rmse


def test_features(df, target_col, k=None, features=None, univariate=True,
                  key=None):
    rmse_results = {}
    if features is None:
        features = df.columns.drop(target_col)
    training_df, test_df = divide_train_test(df)
    # if no k value provided set to default 5
    if k == None:
        k = [5]
    k_rmses = {}
    if univariate:
        # we model the provided features one by one
        for feature in features:
            key_dict = feature
            if key != None:
                key_dict = key
            k_rmses = {}
            for k_val in k:
                k_rmses[k_val] = knn_train_test(df,
                                                training_df, feature,
                                                test_df, target_col, k_val)

            rmse_results[key_dict] = k_rmses
        return k, rmse_results
    else:
        # multivariate. We will model with all the provided features
        k_rmses = {}
        for k_val in k:
            k_rmses[k_val] = knn_train_test(df,
                                            training_df, features,
                                            test_df, target_col, k_val)

            key_dict = ",".join(features)
            if key != None:
                key_dict = key
            rmse_results[key_dict] = k_rmses
        return k, rmse_results


def test_multivariate(k=[5]):
    # we will test multiple columns together here
    k_rmse_results_multi = {}
    two_best_features = ['horsepower', 'width']
    k, k_rmse_results = test_features(numeric_cars, "price",
                                      features=two_best_features,
                                      univariate=False,
                                      key='two best features', k=k)
    k_rmse_results_multi.update(k_rmse_results)

    three_best_features = ['horsepower', 'width', 'curb-weight']
    k, k_rmse_results = test_features(numeric_cars, "price",
                                      features=three_best_features,
                                      univariate=False,
                                      key='three best features', k=k)
    k_rmse_results_multi.update(k_rmse_results)

    four_best_features = ['horsepower', 'width', 'curb-weight', 'city-mpg']
    k, k_rmse_results = test_features(numeric_cars, "price",
                                      features=four_best_features,
                                      univariate=False,
                                      key='four best features', k=k)
    k_rmse_results_multi.update(k_rmse_results)

    five_best_features = ['horsepower', 'width', 'curb-weight',
                          'city-mpg', 'highway-mpg']
    k, k_rmse_results = test_features(numeric_cars, "price",
                                      features=five_best_features,
                                      univariate=False,
                                      key='five best features', k=k)
    k_rmse_results_multi.update(k_rmse_results)

    six_best_features = ['horsepower', 'width', 'curb-weight',
                         'city-mpg', 'highway-mpg', 'length']
    k, k_rmse_results = test_features(numeric_cars, "price",
                                      features=six_best_features,
                                      univariate=False,
                                      key='six best features', k=k)
    k_rmse_results_multi.update(k_rmse_results)

    return k_rmse_results_multi


def test_univariate(numeric_cars):
    # test univarate features with default k = 5
    k, rmse_results = test_features(numeric_cars, "price")
    print(rmse_results)
    print()

    # test univarate features with list of ks
    k = [1, 3, 5, 7, 9]
    k, k_rmse_results = test_features(numeric_cars, "price", k=k)
    print(k_rmse_results)
    print()
    return k, k_rmse_results


def plot_univariate(k_rmse_results_uni):
    for key in k_rmse_results_uni:
        v = k_rmse_results_uni[key]
        x = sorted(v)
        y = list(v.values())
        plt.plot(x, y, label=key)
    plt.xlabel('k value')
    plt.ylabel('RMSE')
    plt.show()


def plot_multivariate(k_rmse_results_multi):
    for k, v in k_rmse_results_multi.items():
        x = list(v.keys())
        y = list(v.values())
        plt.plot(x, y, label=k)
        plt.xlabel('k value')
        plt.ylabel('RMSE')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':

    # read df
    numeric_cars = get_car_df()
    # clean df
    numeric_cars = clean_df(numeric_cars)

    # test univariate
    k, k_rmse_results_uni = test_univariate(numeric_cars)
    # plot univariate
    plot_univariate(k_rmse_results_uni)

    feature_avg_rmse = get_feature_avg_rmse(k_rmse_results_uni)
    series_avg_rmse = pd.Series(feature_avg_rmse)
    print(series_avg_rmse.sort_values())
    print()

    # test multivariate with default k
    k_rmse_results_multi = test_multivariate()
    print(k_rmse_results_multi)
    print()

    # generate k value from 1 to 25 and call test_multivariate with k list
    k = [i for i in range(1, 25)]
    k_rmse_results_multi = test_multivariate(k)
    print(k_rmse_results_multi)
    print()

    plot_multivariate(k_rmse_results_multi)
