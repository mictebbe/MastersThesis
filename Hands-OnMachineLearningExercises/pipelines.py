from load_housing_data import load_data, print_info
from create_test_set_final import split_housing_data
from custom_transformers import CombinedAttributesAdder, DataFrameSelector, MyLabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, Imputer


SEED = 42
housing = load_data()

train_set,test_set = split_housing_data(housing,SEED)
housing_num = train_set.drop("ocean_proximity",axis=1)

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('data_frame_selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder(add_bedrooms_per_room = True)),
    ('std_scaler', StandardScaler())
    ])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', MyLabelBinarizer())
    ])
full_pipeline = FeatureUnion(transformer_list = [
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])
#housing_num_tr = num_pipeline.fit_transform(housing)
#housing_cat_tr = cat_pipeline.fit_transform(housing)


#print(housing_num_tr)
#print(housing_cat_tr)
def prepare_housing_data(data):
    return full_pipeline.fit_transform(data)
#housing_prepared = prepare_housing_data(train_set)

#print (housing_prepared[0])
#print (housing_prepared.shape)

