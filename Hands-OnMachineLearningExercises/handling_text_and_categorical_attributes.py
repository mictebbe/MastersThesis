from load_housing_data import load_data, print_info
from create_test_set_final import split_housing_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import Imputer
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer

SEED = 42
housing = load_data()
train_set,test_set = split_housing_data(housing,SEED)

label_encoder = LabelEncoder()

housing_cat = housing["ocean_proximity"]
#Encodes as Values 0-n for categories
housing_cat_encoded = label_encoder.fit_transform(housing_cat)

print(housing_cat_encoded)
print(label_encoder.classes_)

#Encodes values as one hot vector from previously numerically encoded categories
one_hot_encoder = OneHotEncoder()
housing_cat_1hot = one_hot_encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
print(housing_cat_1hot.toarray())

#Encodes values as one hot vector from Categories
label_binarizer_encoder=LabelBinarizer()
housing_cat_1hot_lb = label_binarizer_encoder.fit_transform(housing_cat)
print( housing_cat_1hot_lb)