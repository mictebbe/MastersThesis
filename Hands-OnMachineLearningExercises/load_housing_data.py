import pandas as  pd
import os
import matplotlib.pyplot as plt

HOUSING_PATH = "datasets/housing"

def load_data(housing_path =HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def print_info(data):
    print("Column info" , data.info()) # pylint: disable=E1101
    print("First 10 rows", data.head(10)) # pylint: disable=E1101
    #print("Value occurences in Column", housing["ocean_proximity"].value_counts())
    print("Summary of numerical attributes", data.describe()) # pylint: disable=E1101
    data.hist(bins=50, figsize=(10,7)) # pylint: disable=E1101
    plt.show()

#housing = load_housing_data()
#print_info(housing)