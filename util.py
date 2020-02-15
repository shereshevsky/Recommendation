import pandas as pd


def get_data():
    train_set_path = 'resources//train_numerized_with_anon.csv'
    test_set_path = 'resources//test_numerized_with_anon.csv'

    train_set = pd.read_csv(train_set_path, parse_dates=[3], index_col='index')
    test_set = pd.read_csv(test_set_path, parse_dates=[3], index_col='index')

    users_in_train = train_set.userID.unique()
    test_set = test_set[test_set.userID.isin(users_in_train)]
    return train_set, test_set
