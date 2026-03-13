import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
Dataset = pd.read_csv("Data.csv")
x = Dataset.iloc[:,:-1].values
y = Dataset.iloc[:,-1].values
print(x)
print(y)

#Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)