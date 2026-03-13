import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
Dataset = pd.read_csv("Data.csv")
x = Dataset.iloc[:,:-1].values
y = Dataset.iloc[:,-1].values
print(x)
print(y)