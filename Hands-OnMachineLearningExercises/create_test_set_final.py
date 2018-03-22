import load_housing_data
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

def split_housing_data(data, seed = 42):
    data["income_cat"] = np.ceil(data["median_income"]/1.5)
    data["income_cat"].where(data["income_cat"] < 5, 5.0, inplace= True)
    split_ = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state=seed)
    for train_index, test_index in split_.split(data, data["income_cat"]):
        train_set_ = data.loc[train_index] # pylint: disable=E1101
        test_set_ = data.loc[test_index] # pylint: disable=E1101
    for set_ in (train_set_, test_set_):
        set_.drop("income_cat", axis=1, inplace=True)
    return train_set_,test_set_