#%%
from worclipo.load_data import load_data

# General packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, SelectFromModel
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from scipy.stats import loguniform, randint

# Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier

#%% Loading the data and separating the test set from the train set
data = load_data()
x = data.drop(["label"], axis='columns').values
y = data["label"].map({'lipoma':0, 'liposarcoma':1})
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42, stratify=y) # Stratified test split, random state for reproducability (42 can be any number), 85/15


#%% Preprocessing (Broken feature removal & Robust scaling)
# Preparation
feature_names = data.drop(["label"], axis=1).columns
x_train_df = pd.DataFrame(x_train, columns=feature_names)
x_test_df = pd.DataFrame(x_test, columns=feature_names)

# Identify 'broken features'
Q1_raw = x_train_df.quantile(0.25)
Q3_raw = x_train_df.quantile(0.75)
IQR_raw = Q3_raw - Q1_raw
broken_mask = (x_train_df - Q3_raw).max() > (1e12) # Find 'broken features' with a relative threshold
broken_features = broken_mask[broken_mask].index.tolist()

# Remove the 'broken features' from both datasets
x_train_df = x_train_df.drop(columns=broken_features)
x_test_df = x_test_df.drop(columns=broken_features)

# Normalize with RobustScaler
scaler = RobustScaler()

# FIT: Scaler learns the median and IQR from traindata
# TRANSFORM: Data is being scaled to this learned information
x_train_scaled = scaler.fit_transform(x_train_df)

# TRANSFORM: Testdata is scaled with the same info from traindata, no info learned from testdata! (no fit_transform here!)
x_test_scaled = scaler.transform(x_test_df)

# Optioneel: Zet ze terug in een DataFrame om kolomnamen te behouden
x_train_final = pd.DataFrame(x_train_scaled, columns=x_train_df.columns)
x_test_final = pd.DataFrame(x_test_scaled, columns=x_test_df.columns)

# Flat feature removal (low variance filtering)
selector = VarianceThreshold(threshold=0.01) # 99% of values for the feature the same

# Fit on scaled data
x_train_selected = selector.fit_transform(x_train_final)

# Apply to test data (no fitting)
x_test_selected = selector.transform(x_test_final)

# Retrieve names of features kept after preprocessing
selected_features = x_train_final.columns[selector.get_support()]

# Convert back to DataFrame for overview
x_train_final = pd.DataFrame(x_train_selected, columns=selected_features)
x_test_final = pd.DataFrame(x_test_selected, columns=selected_features)

print(f"Preprocessing and VarianceThreshold completed: {len(selected_features)} features remaining.")


#%% Search Preparation

rng = np.random.RandomState(42) # fixed seed for reproducability
n_fs_samples = 20 # how many random feature selectors to generate per type
n_features = x_train_final.shape[1]

# T-test feature selection
k_min = int(0.01 * n_features) # Lower bound, 
k_max = int(0.15 * n_features) # Upper bound, rule of thumb is n_features < n_samples / 10
k_samples = rng.randint(k_min, k_max, size=n_fs_samples) # randomly generate k values within defined range (random search)
kbest_options = [SelectKBest(f_classif, k=int(k)) for k in k_samples] # f_classif = T-test, loop for different k's, randomized search

# LASSO feature selection
c_samples = loguniform.rvs(1000, 100000, size=n_fs_samples, random_state=rng) # Because LASSO reciproke works logarithmic, loguniform is needed to equally distribute random generated c's over the whole range
lasso_options = [
    SelectFromModel(
        LogisticRegression( # l1_ratio=1.0 makes l1 penalty: sets feature weights to 0, max_iterations to perform well, solver saga because 0<=c<=1. Max iterations to allow convergence, tolerance also helps
            solver='saga', 
            l1_ratio=1.0, 
            C=float(c), 
            max_iter=5000, 
            tol=1e-3
            ),
            max_features=max(1, int(0.1 * n_features))  # always select at least 10% of features (otherwise, LogReg sets all features to zero)

    )
    for c in c_samples
    ]

# PCA
n_components_samples = rng.randint(2, 50, size=n_fs_samples)
pca_options = [PCA(n_components=int(n)) for n in n_components_samples]

all_fs_options = kbest_options + pca_options + ['passthrough'] #+lasso_options # pass-through makes no extra feature selection, only variance threshold of preprocessing is applied


recon_clfs = [
    LinearDiscriminantAnalysis(solver="eigen", shrinkage=True),
    QuadraticDiscriminantAnalysis(solver="eigen", shrinkage=True),
    GaussianNB(),
    LogisticRegression(max_iter=1000),
    SGDClassifier(),
    SVC(),
    NuSVC(),
    DecisionTreeClassifier(),
    KNeighborsClassifier(),
    RandomForestClassifier(), 
    GradientBoostingClassifier(), 
    AdaBoostClassifier()
    ]

# %%
scoring = {
    'accuracy': metrics.make_scorer(metrics.accuracy_score),
    'auc': metrics.make_scorer(metrics.roc_auc_score),
    'f1': metrics.make_scorer(metrics.f1_score),
    'precision': metrics.make_scorer(metrics.precision_score),
    'recall': metrics.make_scorer(metrics.recall_score)
}

recon_results = {}

for clf in recon_clfs:
    clf_type = type(clf)

    pipe = Pipeline([
        ('fs', 'passthrough'), # placeholder, will be overwritten by RandomizedSearchCV
        ('clf', clf)
    ])

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # 5-fold, shuffles the order of the data before splitting, with reproducability
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions={'fs': all_fs_options},
        n_iter=30, # Nr. of times repeated
        cv=inner_cv,
        scoring='roc_auc', # roc_auc is the most complete metric, which includes both specificity and sensitivity, making it the best for hyperparameter optimalization
        random_state=42,
        error_score=0    
        )

    cv_results = cross_validate(
        search,
        x_train_final, y=y_train,
        cv=outer_cv,
        scoring=scoring,
        return_train_score=False,   # Not relevant
    )

    recon_results[type(clf).__name__] = {
        metric.replace('test_', ''): (cv_results[metric].mean(), cv_results[metric].std())
        for metric in ['test_accuracy', 'test_auc', 'test_f1', 'test_precision', 'test_recall']
    }
#%%
# Print ranked by AUC
print("Reconnaissance CV results (ranked by AUC):")
print("-" * 50)
sorted_results = sorted(recon_results.items(), key=lambda x: x[1]['auc'][0], reverse=True)
for name, metrics_dict in sorted_results:
    print(f"\n{name}")
    for metric, (mean, std) in metrics_dict.items():
        print(f"  {metric}: {mean:.3f} ± {std:.3f}")
# %%
