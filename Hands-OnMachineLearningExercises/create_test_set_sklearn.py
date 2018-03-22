import load_housing_data
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
housing = load_housing_data.load_housing_data()
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace= True)

#load_housing_data.print_info(housing)
SEED = 42

###Random Split###
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=SEED)
print (len(train_set), "train +", len(test_set), "test")

#load_housing_data.print_info(train_set)

###Stratfied Split###
#ensures representative test_set, train_set


split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state=SEED)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index] # pylint: disable=E1101
    strat_test_set = housing.loc[test_index] # pylint: disable=E1101

def print_split_comparison():
    print("Income categories proportions:")
    print("full set", housing["income_cat"].value_counts() / len(housing))
    print("test set random", train_set["income_cat"].value_counts() / len(train_set))
    print("train set random", test_set["income_cat"].value_counts() / len(test_set))
    print("test set stratified", strat_train_set["income_cat"].value_counts() / len(strat_train_set))
    print("train set stratified", strat_test_set["income_cat"].value_counts() / len(strat_test_set))

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

def split_housing_data(data):
    split_ = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state=SEED)
    for train_index, test_index in split_.split(housing, housing["income_cat"]):
        train_set_ = housing.loc[train_index] # pylint: disable=E1101
        test_set_ = housing.loc[test_index] # pylint: disable=E1101
    for set_ in (train_set_, test_set_):
        set_.drop("income_cat", axis=1, inplace=True)
    return train_set_,test_set_