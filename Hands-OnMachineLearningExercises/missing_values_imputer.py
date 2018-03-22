from load_housing_data import load_data, print_info
from create_test_set_final import split_housing_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import Imputer
import pandas as pd
SEED = 42
housing = load_data()
train_set,test_set = split_housing_data(housing,SEED)

housing = train_set.drop("median_house_value",axis=1)
housing_labels = train_set["median_house_value"].copy()

imputer = Imputer(strategy="median")

housing_num = housing.drop("ocean_proximity",axis=1)

imputer.fit(housing_num)

print ("imputer statistics", imputer.statistics_)
print ("Housing median values", housing_num.median().values)

#returns np.array()
x=imputer.transform(housing_num)

housing_tr= pd.DataFrame(x, columns=housing_num.columns)
print_info(housing_tr)