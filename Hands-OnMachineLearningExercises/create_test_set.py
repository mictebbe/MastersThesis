import load_housing_data
import numpy as np
housing = load_housing_data.load_housing_data()
#load_housing_data.print_info(housing)
SEED = 42

#should be done with hashes of unique identifiers if data frquently changes (see p. 50)
def split_train_test(data, test_ratio):
    np.random.seed(SEED) # pylint: disable=E1101
    shuffled_indices = np.random.permutation(len(data)) # pylint: disable=E1101
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print (len(train_set), "train +", len(test_set), "test")

load_housing_data.print_info(train_set)