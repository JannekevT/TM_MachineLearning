# normaliseren van data voor XGBoost, Random forest met RobustScaler
# wel nog verwijderde outliers, maar deze worden niet meer geclipped
# RobustScaler is minder gevoelig voor outliers, maar we willen ze nog steeds verwijderen


from worclipo.load_data import load_data
from sklearn.model_selection import train_test_split


# General packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets as ds
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler


data = load_data()

#split data into train and test sets
x = data.drop(["label"], axis='columns').values
y = data["label"].map({'lipoma':0, 'liposarcoma':1})
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42, stratify = y)

# --- 1. Voorbereiding ---
feature_names = data.drop(["label"], axis=1).columns

x_train_df = pd.DataFrame(x_train, columns=feature_names)
x_test_df = pd.DataFrame(x_test, columns=feature_names)

# --- 2. Identificeer Kapotte Features ---
# We doen dit op de ruwe x_train_df
Q1_raw = x_train_df.quantile(0.25)
Q3_raw = x_train_df.quantile(0.75)
IQR_raw = Q3_raw - Q1_raw

# Gebruik een relatieve threshold om 'kapotte' features te vinden
broken_mask = (x_train_df - Q3_raw).max() > (1e12)
broken_features = broken_mask[broken_mask].index.tolist()

print(f"Aantal verwijderde 'kapotte' features: {len(broken_features)}")

# Verwijder ze direct uit beide sets
x_train_df = x_train_df.drop(columns=broken_features)
x_test_df = x_test_df.drop(columns=broken_features)

from sklearn.preprocessing import RobustScaler

# --- 3. Normaliseren met RobustScaler ---
# Initialiseer de scaler
scaler = RobustScaler()

# FIT: De scaler leert de mediaan en IQR van de TRAIN data
# TRANSFORM: De data wordt daadwerkelijk geschaald
x_train_scaled = scaler.fit_transform(x_train_df)

# TRANSFORM: Pas dezelfde schaal toe op de TEST data (geen fit_transform hier!)
x_test_scaled = scaler.transform(x_test_df)

# Optioneel: Zet ze terug in een DataFrame om kolomnamen te behouden
x_train_final = pd.DataFrame(x_train_scaled, columns=x_train_df.columns)
x_test_final = pd.DataFrame(x_test_scaled, columns=x_test_df.columns)

# print(f"Data genormaliseerd. Shape van train set: {x_train_final.shape}")
# print(f"Shape van test set: {x_test_final.shape}")

# platte features weghalen - Verwijder features met (bijna) geen variatie ---
# We stellen een drempelwaarde in. 
# Bij '0' verwijder je alleen features die voor iedereen EXACT hetzelfde zijn.
# Bij '0.01' verwijder je features waar 99% van de data hetzelfde is.
selector = VarianceThreshold(threshold=0.01) 

# Fit op de geschaalde train data
x_train_selected = selector.fit_transform(x_train_final)

# Pas toe op de test data
x_test_selected = selector.transform(x_test_final)

# Haal de namen op van de features die we OVERHOUDEN
selected_features = x_train_final.columns[selector.get_support()]

# Zet ze weer terug in een DataFrame voor het overzicht
x_train_final = pd.DataFrame(x_train_selected, columns=selected_features)
x_test_final = pd.DataFrame(x_test_selected, columns=selected_features)

print(f"VarianceThreshold voltooid: {len(selected_features)} features overgebleven.")
