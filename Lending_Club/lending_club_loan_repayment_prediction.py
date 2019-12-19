import pandas as pd
import os


def read_df(filename):
    dir_name = os.path.dirname(os.path.abspath(__file__))
    full_file_name = os.path.join(dir_name, filename)
    df = pd.read_csv(full_file_name)
    print("\nShape after reading - ", df.shape, "#"*20)
    return df


def drop_duplicates_and_useless_columns(df):
     # remokve duplicates
    df.drop_duplicates()
    # remove columns which leaks information from the future to  the model - set 1
    # All of these columns leak data from the future, meaning that they're describing aspects of the loan
    # after it's already been fully funded and started to be paid off by the borrower.
    df = df.drop(["id", "member_id", "funded_amnt", "funded_amnt_inv",
                  "grade", "sub_grade", "emp_title", "issue_d"], axis=1)
    # remove columns which leaks information from the future to  the model - set 2
    df = df.drop(["zip_code", "out_prncp", "out_prncp_inv",
                  "total_pymnt", "total_pymnt_inv", "total_rec_prncp"], axis=1)
    # remove columns which leaks information from the future to  the model - set 3
    df = df.drop(["total_rec_int", "total_rec_late_fee", "recoveries",
                  "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt"], axis=1)
    return df


def clean_target_column(df):
     # now lets concentrate on the target column 'loan_status'
    print()
    print(loans_2007['loan_status'].value_counts())
    print()
    # since there are mulitple values in the column and we are primarily looking to predict whether loan was paid
    # off or charged off, lets go ahead nad remvoe all the rows which don't have these values
    df = df[(df['loan_status'] == "Fully Paid") |
            (df['loan_status'] == "Charged Off")]
    # we will now change the target column value into binary values
    loan_status_replace = {
        "loan_status": {
            "Fully Paid": 1,
            "Charged Off": 0,
        }
    }
    df = df.replace(loan_status_replace)
    return df


def remove_columns_with_one_unique_values(df):
    # now lets remove all the columns which have only one unique value. columsn with only one unique value does not add
    # information for our model
    columns_with_one_unique_values = []
    for column in df.columns:
        column_vals = df[column].dropna().unique()
        if(len(column_vals) <= 1):
            columns_with_one_unique_values.append(column)
    print("\ndropping the following columns which have one or less unique values",
          columns_with_one_unique_values, "\n")
    df = df.drop(columns_with_one_unique_values, axis=1)
    return df


def clean_df(df):
    df = drop_duplicates_and_useless_columns(df)
    df = clean_target_column(df)
    df = remove_columns_with_one_unique_values(df)
    print("\nShape after cleaning df - ", df.shape, "#"*20)
    return df


def feature_engineering(df):
    null_counts = df.isnull().sum()
    print("\nnull value counts in df...")
    print(null_counts[null_counts > 0], "\n")
    # This means that we'll keep the following columns and just remove rows containing missing
    # values for them: emp_length title revol_util last_credit_pull_d and remove column pub_rec_bankruptcies
    df = df.drop("pub_rec_bankruptcies", axis=1)
    df = df.dropna(axis=0)
    # print(df.dtypes.value_counts(dropna=False), "\n")

    # now lots of columns seems to hav categorical data. The column type is object. So lets explore those columns
    object_columns_df = df.select_dtypes(include=['object'])
    # print("\n", object_columns_df.head(1), "\n")

    # potential categorical columns
    cols = ['home_ownership', 'verification_status',
            'emp_length', 'term', 'addr_state', 'title', 'purpose']
    print("\nfollowing are potential categorical columns - ", cols, "\n")
    # for column in cols:
    #     print("\ncolumn = '", column, "'\n", df[column].value_counts())

    # we are now ready to drop columns which will not add values. We will also map emp_length to numerical values
    # dict for replacing numercal values in emp_lenght column
    mapping_dict = {
        "emp_length": {
            "10+ years": 10,
            "9 years": 9,
            "8 years": 8,
            "7 years": 7,
            "6 years": 6,
            "5 years": 5,
            "4 years": 4,
            "3 years": 3,
            "2 years": 2,
            "1 year": 1,
            "< 1 year": 0,
            "n/a": 0
        }
    }
    # drp column which are not adding value
    df = df.drop(["last_credit_pull_d", "earliest_cr_line",
                  "addr_state", "title"], axis=1)
    df["int_rate"] = df["int_rate"].str.rstrip("%").astype("float")
    df["revol_util"] = df["revol_util"].str.rstrip("%").astype("float")
    # replace values in emp_length
    df = df.replace(mapping_dict)

    # now we are ready to create dummy columns. First get the dummy, add it back to df and then remove original columns
    cat_columns = ["home_ownership", "verification_status", "purpose", "term"]
    print("\nconverting the following columns which have categorical values to dummy columns", cat_columns, "\n")
    dummy_df = pd.get_dummies(df[cat_columns])
    df = pd.concat([df, dummy_df], axis=1)
    df = df.drop(cat_columns, axis=1)
    print("\nShape after feature engineering - ", df.shape, "#"*20)

    return df


def get_features_target(df):
    cols = df.columns
    train_cols = cols.drop("loan_status")
    features = df[train_cols]
    target = df["loan_status"]
    return features, target


def test_and_validate_lr(features, target):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    # our dataset is very unbalanced as paid is 6 times more likely than default
    penalty = {
        0: 6,
        1: 1
    }
    lr = LogisticRegression(class_weight=penalty, max_iter=500)
    predictions = pd.Series(cross_val_predict(lr, features, target, cv=3))
    tpr, fpr = calculate_tpr_fpr(predictions)
    return tpr, fpr


def test_and_validate_random_forest(features, target):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    penalty = {
        0: 6,
        1: 1
    }
    rf = RandomForestClassifier(class_weight=penalty, random_state=1)
    predictions = pd.Series(cross_val_predict(rf, features, target, cv=3))
    tpr, fpr = calculate_tpr_fpr(predictions)
    return tpr, fpr


def calculate_tpr_fpr(predictions):

    fp = len(predictions[((predictions == 1) & (target == 0))])
    tp = len(predictions[((predictions == 1) & (target == 1))])
    fn = len(predictions[((predictions == 0) & (target == 1))])
    tn = len(predictions[((predictions == 0) & (target == 0))])
    fpr = fp/(fp+tn)
    tpr = tp/(tp + fn)

    return tpr, fpr


if __name__ == '__main__':
    loans_2007 = read_df("loans_2007.csv")
    loans_2007 = clean_df(loans_2007)
    loans_2007 = feature_engineering(loans_2007)
    features, target = get_features_target(loans_2007)
    tpr, fpr = test_and_validate_lr(features, target)
    print("#"*30, " LogisticRegression results ", "#"*30)
    print("\nfalse positive rate = {:0.4f} and true positive rate = {:0.4f}\n".format(
        fpr, tpr))
    tpr, fpr = test_and_validate_random_forest(features, target)
    print("#"*30, " RandomForestClassifier results ", "#"*30)
    print("\nfalse positive rate = {:0.4f} and true positive rate = {:0.4f}\n".format(
        fpr, tpr))
