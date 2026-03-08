#normaliseren van de data met clip en MinMaxScaler
#voor K-NN model of iets dat gevoelig is voor outliers


#%%
from worclipo.load_data import load_data
from sklearn.model_selection import train_test_split

# General packages
import numpy as np
import pandas as pd
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

# --- 3. Outlier Capping (Mediaan) ---
# Bereken nieuwe statistieken op de overgebleven features
Q1 = x_train_df.quantile(0.25)
Q3 = x_train_df.quantile(0.75)
IQR = Q3 - Q1
train_median = x_train_df.median()

# Definieer de grenzen
lower_limit = Q1 - 1.5 * IQR 
upper_limit = Q3 + 1.5 * IQR 

# Gebruik de krachtige .clip() methode voor snelheid (ipv een loop met .loc)
# Let op: .clip() vervangt uitschieters door de grens (Winsorizing). 
# Wil je echt de mediaan? Houd dan je loop, maar .clip is vaak stabieler voor modellen.
x_train_clean = x_train_df.clip(lower_limit, upper_limit, axis=1)
x_test_clean = x_test_df.clip(lower_limit, upper_limit, axis=1)

# # Als je toch perse de mediaan wilt ipv de grens, gebruik je loop maar optimaliseer deze:
# for col in x_train_df.columns:
#     # Train
#     mask_tr = (x_train_df[col] < lower_limit[col]) | (x_train_df[col] > upper_limit[col])
#     x_train_df.loc[mask_tr, col] = train_median[col]
#     # Test
#     mask_ts = (x_test_df[col] < lower_limit[col]) | (x_test_df[col] > upper_limit[col])
#     x_test_df.loc[mask_ts, col] = train_median[col]

# print("Klaar! Outliers zijn behandeld.")

scaler = MinMaxScaler()
# Fit op de 'geclipte' train data
x_train_scaled = scaler.fit_transform(x_train_clean)
# Pas toe op de 'geclipte' test data
x_test_scaled = scaler.transform(x_test_clean)

# Optioneel: Zet ze terug in een DataFrame om kolomnamen te behouden
x_train_final = pd.DataFrame(x_train_scaled, columns=x_train_df.columns)
x_test_final = pd.DataFrame(x_test_scaled, columns=x_test_df.columns)

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
