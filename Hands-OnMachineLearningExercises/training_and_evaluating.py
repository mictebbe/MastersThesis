from load_housing_data import load_data, print_info
from create_test_set_final import split_housing_data
from pipelines import prepare_housing_data
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
import pandas as pd
from sklearn.linear_model import LinearRegression

SEED = 42
housing = load_data()

train_set,test_set = split_housing_data(housing,SEED)
housing_prepared = prepare_housing_data(train_set)
housing_labels = train_set["median_house_value"].copy()
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


some_data = train_set.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = prepare_housing_data(some_data)

print ("Predictions:", list(lin_reg.predict(some_data_prepared)))
#print ("Labels:", list(some_labels))
#print ("Labels:", list(some_data_prepared))