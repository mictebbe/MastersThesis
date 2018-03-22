from load_housing_data import load_data
from create_test_set_final import split_housing_data
import matplotlib.pyplot as plt
import numpy as np

SEED = 42
housing = load_data()
train_set,test_set = split_housing_data(housing,SEED)

housing = train_set.drop("median_house_value",axis=1)
housing_labels = train_set["median_house_value"].copy()

#Cleaning Data
#Option 1 Drop corresponding Values
dropped_values_set = train_set.dropna(subset=["total_bedrooms"])

#Option 2 Drop Column
dropped_column_set = train_set.drop("total_bedrooms",axis=1)

#Option 3 Fill with median Value
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace = True)

