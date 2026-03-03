# ## Data loading and cleaning
# Below are functions to load the dataset of your choice. After that, it is all up to you to create and evaluate a classification method. Beware, there may be missing values in these datasets. Good luck!


#%% Data loading functions. Uncomment the one you want to use
#from worcgist.load_data import load_data
from worclipo.load_data import load_data
#from worcliver.load_data import load_data
#from hn.load_data import load_data
#from ecg.load_data import load_data
from sklearn.model_selection import train_test_split
# General packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets as ds
import seaborn

# Classifiers
from sklearn import model_selection
from sklearn import metrics


data = load_data()
print(f'The number of samples: {len(data.index)}')

print(f'The number of columns: {len(data.columns)}')

#number of lipoma, column 2 = lipoma
print(f'The number of lipoma: {(data["label"] == "lipoma").sum()}')
#number of liposarcoma, column 2 = liposarcoma
print(f'The number of liposarcoma: {(data["label"] == "liposarcoma").sum()}')

#number of missing values
print("Number of total missing values:", data.isnull().sum().sum())
print("Number of feature missing values:", data.iloc[:, 2:].isnull().sum().sum())

#split data into train and test sets
x = data.drop(["label"], axis='columns').values
y = data["label"].map({'lipoma':0, 'liposarcoma':1})
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42, stratify = y)

#plot data
fig = plt.figure(figsize=(24,8))

ax = fig.add_subplot(131)
ax.set_title("entire dataset", fontsize='small')
ax.scatter(x[:, 0], x[:, 1], marker='o', c=y,
           s=25, edgecolor='k', cmap=plt.cm.Paired)
ax = fig.add_subplot(132)
ax.set_title("Training data", fontsize='small')
ax.scatter(x_train[:, 0], x_train[:, 1], marker='o', c=y_train,
           s=25, edgecolor='k', cmap=plt.cm.Paired)

ax = fig.add_subplot(133)
ax.set_title("Test data", fontsize='small')
ax.scatter(x_test[:, 0], x_test[:, 1], marker='o', c=y_test,
           s=25, edgecolor='k', cmap=plt.cm.Paired)

