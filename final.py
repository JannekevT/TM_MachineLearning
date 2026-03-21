#%% Packages
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, SelectFromModel
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from worclipo.load_data import load_data

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


clfs = [
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    KNeighborsClassifier()
]

param_grids = {
    RandomForestClassifier: {
        'fs': all_fs_options,
        'clf__n_estimators': randint(50, 200),
        'clf__max_depth': randint(2, 6),
        'clf__min_samples_split': randint(5, 20),
    },
    GradientBoostingClassifier: {
        'fs': all_fs_options,
        'clf__n_estimators': randint(50, 200),
        'clf__learning_rate': loguniform(0.01, 0.2),
        'clf__max_depth': randint(2, 6),
    },
    KNeighborsClassifier: {
        'fs': all_fs_options,
        'clf__n_neighbors': randint(3, 20),
        'clf__weights': ['uniform', 'distance'],
    },
}

#%%
# Nested Cross-validation Loop
scoring = {
    'accuracy': metrics.make_scorer(metrics.accuracy_score),
    'auc': metrics.make_scorer(metrics.roc_auc_score),
    'f1': metrics.make_scorer(metrics.f1_score),
    'precision': metrics.make_scorer(metrics.precision_score),
    'recall': metrics.make_scorer(metrics.recall_score)
}

results = {}

for clf in clfs:
    clf_type = type(clf)
    param_grid = param_grids[clf_type]

    pipe = Pipeline([
        ('fs', 'passthrough'), # placeholder, will be overwritten by RandomizedSearchCV
        ('clf', clf)
    ])

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # 5-fold, shuffles the order of the data before splitting, with reproducability
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        n_iter=100, # Nr. of times repeated
        cv=inner_cv,
        scoring='roc_auc', # roc_auc is the most complete metric, which includes both specificity and sensitivity, making it the best for hyperparameter optimalization
        random_state=42,
        error_score=0 # if LASSO produces 0 features, score 0 rather than crash
    )

    cv_results = cross_validate(
        search,
        x_train_final, y=y_train,
        cv=outer_cv,
        scoring=scoring,
        return_train_score=False,   # Not relevant
    )

    results[clf_type.__name__] = cv_results
    print(clf_type.__name__)
    for metric in ['test_accuracy', 'test_auc', 'test_f1', 'test_precision', 'test_recall']:
        mean = cv_results[metric].mean()
        std = cv_results[metric].std()
        print(f"  {metric.replace('test_', '')}: {mean:.3f} ± {std:.3f}")
    print()

#%% Final model training
final_clf = # Best performing model
final_clf_type = type(final_clf)

final_pipe = Pipeline([
    ('fs', 'passthrough'),
    ('clf', final_clf)
])

final_inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

final_search = RandomizedSearchCV(
    estimator=final_pipe,
    param_distributions=param_grids[final_clf_type],
    n_iter=40,
    cv=final_inner_cv,
    scoring='roc_auc',
    refit=True,             # fits best model on full x_train_final after search
    random_state=42,
    error_score=0
)

# Fit on ALL training data
final_search.fit(x_train_final, y_train)
print("Best params:", final_search.best_params_)
print("Best inner CV AUC:", final_search.best_score_)

#%% Final evaluation on test set
y_pred = final_search.predict(x_test_final)
y_prob = final_search.predict_proba(x_test_final)[:,1] # Output is 2 columns, this selects only the column "prob_class_1", not class 0

print("Test AUC:      ", metrics.roc_auc_score(y_test, y_prob))
print("Test Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Test F1:       ", metrics.f1_score(y_test, y_pred))
print("Test Precision:", metrics.precision_score(y_test, y_pred))
print("Test Recall:   ", metrics.recall_score(y_test, y_pred))


""""
The full picture

nested CV  →  tells you expected generalisation performance (AUC ± std)
     ↓
final RandomizedSearchCV on x_train_final  →  finds best hyperparameters
     ↓
final_search.best_estimator_  →  your deployable model
     ↓
evaluate once on x_test_final  →  unbiased final performance estimate"

""""