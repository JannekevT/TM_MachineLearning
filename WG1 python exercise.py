# WG1 python exercise
from os import read
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

## start
df = pd.read_csv("datasets.csv")
num_df = df['dataset'].nunique()
print (num_df)

## nr of datasets
print (df['dataset'].value_counts())    
#names of datasets
print (df['dataset'].unique())  
#statistics per datasets
print (df.groupby('dataset').describe())
#count statistics per dataset
print (df.groupby('dataset').count())
#mean statistics per dataset
print (df.groupby('dataset').mean())
#variance statistics per dataset
print (df.groupby('dataset').var())
#standard deviation statistics per dataset
print (df.groupby('dataset').std())

## violin plots of x-coordinates per dataset next to eachother
#in a subplot with x and y coordinates
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.violinplot(x='dataset', y='x', data=df, ax=axes[0])
axes[0].set_title('Violin Plot of x-coordinates per dataset')
sns.violinplot(x='dataset', y='y', data=df, ax=axes[1])
axes[1].set_title('Violin Plot of y-coordinates per dataset')
plt.tight_layout()
plt.show()  

# determine and print correlation between x and y coordinates for each dataset
for dataset in df['dataset'].unique():
    subset = df[df['dataset'] == dataset]
    correlation = subset['x'].corr(subset['y'])
    print(f"Correlation between x and y for {dataset}: {correlation}")  

# determine and print covariance between x and y coordinates for each dataset
for dataset in df['dataset'].unique():
    subset = df[df['dataset'] == dataset]
    covariance = subset['x'].cov(subset['y'])
    print(f"Covariance between x and y for {dataset}: {covariance}")

# determine linear regression between x and y for each dataset
for dataset in df['dataset'].unique():
    subset = df[df['dataset'] == dataset]
    slope, intercept, r_value, p_value, std_err = stats.linregress(subset['x'], subset['y'])
    print(f"Linear regression for {dataset}: slope={slope}, intercept={intercept}, r_value={r_value}, p_value={p_value}, std_err={std_err}")   
# print slope, intercept and r-value for each dataset
for dataset in df['dataset'].unique():
    subset = df[df['dataset'] == dataset]
    slope, intercept, r_value, p_value, std_err = stats.linregress(subset['x'], subset['y'])
    print(f"Linear regression for {dataset}: slope={slope}, intercept={intercept}, r_value={r_value}")

#create scatter plots for all datasets
for dataset in df['dataset'].unique():
    subset = df[df['dataset'] == dataset]
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x='x', y='y', data=subset)
    plt.title(f'Scatter Plot for {dataset}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# create scatter plots with regression line for all datasets
for dataset in df['dataset'].unique():
    subset = df[df['dataset'] == dataset]
    plt.figure(figsize=(6, 4))
    sns.regplot(x='x', y='y', data=subset)
    plt.title(f'Scatter Plot with Regression Line for {dataset}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


