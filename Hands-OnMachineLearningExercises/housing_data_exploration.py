from load_housing_data import load_data
from create_test_set_final import split_housing_data
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
SEED = 42
housing = load_data()
train_set,test_set = split_housing_data(housing,SEED)

train_set.plot(kind="scatter", x="longitude", y="latitude", alpha = 0.4,
s=housing["population"]/100, label = "population", figsize = (10,7),
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar = True)


corr_matrix = train_set.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]

scatter_matrix(test_set[attributes], figsize=(12,8))

train_set.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

plt.show()

train_set["rooms_per_household"] = train_set["total_rooms"]/train_set["households"]
train_set["bedrooms_per_room"] = train_set["total_bedrooms"]/train_set["total_rooms"]
train_set["population_per_household"] = train_set["population"]/train_set["households"]

corr_matrix_new_attributes = train_set.corr()
print(corr_matrix_new_attributes["median_house_value"].sort_values(ascending=False))
