import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.model_selection import KFold

target = "SalePrice"
corr_threshold = 0.4
unique_threshold = 10
rmse_results = {}

# read df


def read_df(filename="AmesHousing.tsv"):
    dir_name = os.path.dirname(os.path.abspath(__file__))
    full_file_name = os.path.join(dir_name, filename)
    df = pd.read_csv(full_file_name, delimiter="\t")
    return df


# LR model
def model_and_predict(train, test, features, target):
     # model on train
    lr = linear_model.LinearRegression()
    lr.fit(train[features], train[target])
    # test on train
    predictions = lr.predict(test[features])

    # Lets find the rmse
    mse = mean_squared_error(test[target], predictions)
    rmse = np.sqrt(mse)
    return rmse

# Version 1 ##################################################
# set of helper functions which will get updating versions


def transform_features_v1(df):
    print("transforming - version 1...")
    return df


def select_features_v1(df):
    print("selecting features - version 1...")
    return df[["Gr Liv Area", "SalePrice"]]


def train_and_test_v1(df, upto=1460):
    print("train and test - version 1...")
    target = "SalePrice"
    # for the first version, lets use all the numerical columns
    # to train the model
    numeric_df = df.select_dtypes(include=['int64', 'float'])
    # we don't want to include "SalePrice" column in our features as it is our traget
    features = numeric_df.columns.drop(target)
    # divide our df into training and test dfs
    train = df[:upto]
    test = df[upto:]

    rmse = model_and_predict(train, test, features, target)
    return rmse, features


def test_model_v1():
    df = read_df()
    print("testing model - Version 1")
    transform_df = transform_features_v1(df)
    filtered_df = select_features_v1(transform_df)
    rmse, features = train_and_test_v1(filtered_df)
    print("testing complete...")
    print("#"*80)
    print("rmse v1 - NO transformation/NO feature selection = {}".format(rmse))
    rmse_results["rmse v1 - NO transformation/NO feature selection"] = rmse


# Version 2 ##################################################
# perform transformation
# Step 1 - Drop any column with 5% or more missing values


def drop_numerical_columns_with_lots_of_missing_values(df):
    numerical_missing_cols = df.isnull().sum()
    # calculate 5%
    five_percent = df.shape[0]/20
    # get cols with more than 5% missing values
    numerical_cols_5_per_null = numerical_missing_cols[numerical_missing_cols > five_percent].sort_values(
    )
    print("numercal columns which have more than 5% missing values")
    print(numerical_cols_5_per_null.index)
    print("the above columns will be deleted")
    # drop them
    df = df.drop(numerical_cols_5_per_null.index, axis=1)
    print("columns successfully deleted...")
    print()
    return df

# Step 2 - Drop any text columns with one or more missing value


def drop_text_columns_with_missing_values(df):
    text_counts = df.select_dtypes(include=["object"])
    text_counts = text_counts.isnull().sum().sort_values(ascending=False)
    text_missing_values = text_counts[text_counts > 0]
    print("text columns which have missing values and their missing count")
    print(text_missing_values)
    print("these rows will be deleted")
    df = df.drop(text_missing_values.index, axis=1)
    print("rows successfully deleted...")
    print()
    return df
# Step 3 - Numerical Columns - For all the missing values in the columns,
# fill it with the most common value in that column (mode)|


def fill_fixable_numercal_cols_with_mode(df):
    num_missing_columns = df.select_dtypes(include=["int64", "float"])
    num_missing_columns = num_missing_columns.isnull().sum().sort_values()
    fixable_columns = num_missing_columns[(
        num_missing_columns < len(df)/20) & (num_missing_columns > 0)]
    print("numerical columns and count of missing values. These columns can be fixed by replacing missing values with column mode")
    print(fixable_columns)
    replacement_values = df[fixable_columns.index].mode().to_dict(orient="records")[
        00]
    print("columns will be udpated with the following mode values")
    print(replacement_values)
    df = df.fillna(replacement_values)
    print("successfully updated")
    print("\ncount of numerical columns with null")
    print(df.isnull().sum().value_counts())
    print()
    return df

# New features to better represent existing features


def add_new_features(df):
    years_sold = df['Yr Sold'] - df['Year Built']
    print("bad values in years sold column")
    print(years_sold[years_sold < 0])
    years_since_remod = df['Yr Sold'] - df['Year Remod/Add']
    print("bad values in years since remodeled column")
    print(years_since_remod[years_since_remod < 0])
    # add these new columns to df
    # Create new columns
    df['Years Before Sale'] = years_sold
    df['Years Since Remod'] = years_since_remod

    # Drop rows with negative values for both of these new features
    df = df.drop([1702, 2180, 2181], axis=0)

    # No longer need original year columns
    df = df.drop(["Year Built", "Year Remod/Add"], axis=1)
    print("new columns created and rows with bad data deleted successfully")
    print()
    return df
# Drop columns which will not contribute to better prediction


def delete_useless_columns(df):
    print("columns ['PID', 'Order'] are not useful for ML and will be deleted")
    # Drop columns that aren't useful for ML
    df = df.drop(["PID", "Order"], axis=1)
    print("successfully deleted columns ....")

    # Drop columns that leak info about the final sale
    print("columns ['Mo Sold', 'Sale Condition', 'Sale Type', 'Yr Sold'] leak information about final sale. So will be deleted")
    df = df.drop(["Mo Sold", "Sale Condition", "Sale Type", "Yr Sold"], axis=1)
    print("successfully deleted...")
    print()
    return df

# Update transform_features function with all the changes


def transform_features_v2(df):
    print("transforming - version 2...")
    df = drop_numerical_columns_with_lots_of_missing_values(df)
    df = drop_text_columns_with_missing_values(df)
    df = fill_fixable_numercal_cols_with_mode(df)
    df = add_new_features(df)
    return df

# Now lets test the new version after making all feature transformation


def test_model_v2():
    print("testing - Version 2")
    print("reading file...")
    df = read_df()
    # we have an updated transform features function. Use new one
    transform_df = transform_features_v2(df)
    # select features not yet implemented. use old versions
    filtered_df = select_features_v1(transform_df)
    # use V1 train and test
    rmse, features = train_and_test_v1(filtered_df)
    print("test completed")
    print("#"*80)
    print("rmse v2 - After transformation/NO feature selection = {}".format(rmse))
    rmse_results["rmse v2 - After transformation/NO feature selection "] = rmse
    return df, transform_df, filtered_df, rmse


# Version 3 ##################################################
# Feature Selection
# Now that we have cleaned and transformed a lot of the features in the
# data set, it's time to move on to feature selection for numerical
# Numerical columns - feature selection
def select_numerical_columns(df, corr_threshold=corr_threshold):
    # Take int and float columns from transfored df
    numerical_df = df.select_dtypes(include=["int64", "float"])
    print("numerical columns are ")
    print(numerical_df.columns)
    print()
    # calculate absolute correleation between columns and target
    abs_corr_coeffs = numerical_df.corr()[target].abs().sort_values()
    print("absolute correlation with 'SalePrice' column")
    print(abs_corr_coeffs)
    print()
    # drop any columns having less than corr_cutoff(0.4)
    strong_corr = abs_corr_coeffs[abs_corr_coeffs >= corr_threshold]
    print("keeping only following columns")
    print(strong_corr.index)
    print()
    print("dropping following columns wich have less than {} correlation".format(
        corr_threshold))
    weak_corr = abs_corr_coeffs[abs_corr_coeffs < corr_threshold]
    print(weak_corr.index)
    # drop weak correlation columns
    try:
        df = df.drop(weak_corr.index, axis=1)
    except(ValueError):
        pass
    print("columns dropped")
    print()
    return df, strong_corr


# Create and display hitmap
def create_corr_heatmap(df, strong_corr):
    print("Creating heat map")
    corr_matrix = df[strong_corr.index].corr()
    import seaborn as sns
    sns.heatmap(corr_matrix)
    return corr_matrix

# Categorical columns - feature selection


def drop_irrelevant_category_columns(df, unique_threshold=unique_threshold):
    # Create a list of column names from documentation that are *meant* to be categorical
    nominal_features = ["PID", "MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour", "Lot Config", "Neighborhood",
                        "Condition 1", "Condition 2", "Bldg Type", "House Style", "Roof Style", "Roof Matl", "Exterior 1st",
                        "Exterior 2nd", "Mas Vnr Type", "Foundation", "Heating", "Central Air", "Garage Type",
                        "Misc Feature", "Sale Type", "Sale Condition"]
    # check which potential category columns are in df
    transform_cat_cols = [col for col in nominal_features if col in df.columns]
    print("We have following potential category columns")
    print("[", ", ".join(transform_cat_cols), "]")
    print()

    # lets get the unique values for above columns
    print("Unique values in the above columns")
    unique_val_in_ca_cols = df[transform_cat_cols].apply(
        lambda col: len(col.value_counts()))
    unique_val_in_ca_cols = unique_val_in_ca_cols.sort_values()
    print(unique_val_in_ca_cols)
    print()

    # drop any columsn which ave more than max_cat_unique_count (10) unique columns.
    # value 10 is arbitrary. Something we can experiement with
    # remember that every unique value adds a column when we add dummies
    more_unique_cols = unique_val_in_ca_cols[unique_val_in_ca_cols >
                                             unique_threshold].index
    print("dropping following columns which have more than {} unique values".format(
        unique_threshold))
    print(more_unique_cols)
    try:
        df = df.drop(more_unique_cols, axis=1)
    except(ValueError):
        pass
    print("All irrelevant category columns successfully dropped")
    print()
    return df


def covert_category_to_dummy(df):
    text_cols = df.select_dtypes(include=['object'])
    print("following columns will be converted to category columns")
    print(text_cols.columns)
    print()
    for col in text_cols:
        df[col] = df[col].astype("category")

    print("converting category columns to dummy columns and adding back to dataframe")
    try:
        df = pd.concat([
            df,
            pd.get_dummies(df.select_dtypes(include=['category']))
        ], axis=1).drop(text_cols, axis=1)
    except(ValueError):
        pass
    print("All category columns successfully converted to dummy columns")
    print()
    return df

# Create new version of select_feature function

# Now that we have developed feature selection, we can update the
# select_feature function to better reflect the changes


def select_features_v2(df):
    print("Selecting features - Version 2")
    df, strong_corr = select_numerical_columns(df)
    create_corr_heatmap(df, strong_corr)
    df = drop_irrelevant_category_columns(df)
    df = covert_category_to_dummy(df)
    return df

# updte the train and test so that it can handle multiple folds


def train_and_test_v2(df, k=0, upto=1460):
    print("training and testing - Version 2")
    numeric_df = df.select_dtypes(include=['int64', 'float'])
    features = numeric_df.columns.drop(target)
    if k == 0:
        # we use the v1 version of this function to test it
        print("testing with k ==0. We can use old Version 1 function")
        rmse, features = train_and_test_v1(df)
        return rmse
    if k == 1:
        # lets break into two separate test and train randomly
        print("testing with k == 1 using model_and_predict")
        shuffled_df = df.sample(frac=1, )
        # now again we can use the train_and_test_V1 on
        train = shuffled_df[:upto]
        test = shuffled_df[upto:]
        rmse_1 = model_and_predict(train, test, features, target)
        rmse_2 = model_and_predict(test, train, features, target)
        avg_rmse = np.mean([rmse_1, rmse_2])
        return rmse_1, rmse_2, avg_rmse
    else:
        kf = KFold(n_splits=k, shuffle=True)
        rmse_values = []
        for train_index, test_index, in kf.split(df):
            train = df.iloc[train_index]
            test = df.iloc[test_index]
            rmse = model_and_predict(train, test, features, target)
            rmse_values.append(rmse)
        avg_rmse = np.mean(rmse_values)
        return rmse_values, avg_rmse


# ready to test third shiny new version
def test_model_v3():
    print("training and testing - Version 3")
    print("reading file...")
    df = read_df()
    # we have an updated transform features function. Use new one
    transform_df = transform_features_v2(df)

    # we now have updated select_features function
    filtered_df = select_features_v2(transform_df)

    # use V1 train and test
    print("#"*80)
    print("rmse with k = 0 and proper feature selection.")
    rmse = train_and_test_v2(filtered_df)
    print("rmse with k = 0 and proper feature selection = {}".format(rmse))
    rmse_results["rmse with k = 0 and proper feature selection"] = rmse
    print()

    # use V2 train and test sing K =1
    print("#"*80)
    print("rmse with k = 1 and proper feature selection.")
    rmse_1, rmse_2, avg_rmse_1 = train_and_test_v2(filtered_df, k=1)
    print("rmse_1 = {} rmse_2 = {} and avg_rmse = {}".format(
        rmse_1, rmse_2, avg_rmse_1))
    rmse_results["avg_rmse with k = 1 and proper feature selection"] = avg_rmse_1
    print()

    # use V2 train and test sing K =4
    print("#"*80)
    print("rmse with k = 4 and proper feature selection.")
    rmse_values, avg_rmse_4 = train_and_test_v2(filtered_df, k=4)
    print("rmse values = {}".format(rmse_values))
    print("avg rmse = {}".format(avg_rmse_4))
    rmse_results["avg_rmse with k = 4 and proper feature selection"] = avg_rmse_4
    print()

    print("test completed")
    return df, transform_df, filtered_df, rmse


if __name__ == '__main__':
    test_model_v1()
    test_model_v2()
    df, transform_df, filtered_df, rmse = test_model_v3()
    rmse_results_series = pd.Series(rmse_results, index=rmse_results.keys())
    print(rmse_results_series.sort_values())
    plt.show()
