#feature selection and importance
#packages inladen
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# dataset laden
df = pd.read_csv("/Users/fleurleenheer/Desktop/Technical Medicine/MachineLearning/TM_MachineLearning/worclipo/Lipo_radiomicFeatures.csv")

# alleen features (zonder label en ID)
X = df.drop(["ID", "label"], axis=1)
variances = X.var()
print(variances.sort_values())

#selector = VarianceThreshold(threshold=0.01) #verwijdert alle features waarvan de variantie ≤ 0.01

#X_selected = selector.fit_transform(X)

#selected_features = X.columns[selector.get_support()]
#print(selected_features) #Namen van overgebleven features zien
#X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

#print(X_selected_df.head())