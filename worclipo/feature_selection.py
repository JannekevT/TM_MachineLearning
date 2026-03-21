#feature selection and importance

#packages inladen
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# dataset laden
data = pd.read_csv("/Users/fleurleenheer/Desktop/Technical Medicine/MachineLearning/TM_MachineLearning/worclipo/Lipo_radiomicFeatures.csv")

# alleen features (zonder label en ID)
x = data.drop(["label","ID"], axis='columns').values
y = data["label"].map({'lipoma':0, 'liposarcoma':1})
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42, stratify=y) # Stratified test split, random state for reproducability, 85/15

#standardiseren
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)  # let op: alleen transform, niet fit!

# LassoCV kiest automatisch de beste lambda via cross-validatie
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(x_train, y_train)

print(f"Beste lambda: {lasso.alpha_:.4f}")

# Welke features zijn geselecteerd?
coefs = pd.Series(lasso.coef_, index=x.columns)
geselecteerd = coefs[coefs != 0].sort_values(key=abs, ascending=False)

print(f"Aantal geselecteerde features: {len(geselecteerd)}")
print(geselecteerd)

# model evalueren
from sklearn.metrics import mean_squared_error, r2_score

y_pred = lasso.predict(x_test)

print(f"R²: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.3f}")