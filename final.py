#%% Packages

import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from scipy.stats import loguniform, randint
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold, RandomizedSearchCV, train_test_split, RepeatedStratifiedKFold, learning_curve
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, SelectFromModel
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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

print(f"Preprocessing and VarianceThreshold completed: {len(selected_features)} features remaining."

#%% Search Preparation (Preparation of Feature Selections)

rng = np.random.RandomState(42) # fixed seed for reproducability
n_fs_samples = 10 # how many random feature selectors to generate per type
n_features = x_train_final.shape[1]

# T-test feature selection
k_min = int(2) # Lower bound 
k_max = int(20) # Upper bound, rule of thumb is n_features < n_samples / 10 --> 12 or sqrt(115)=11
k_samples = rng.randint(k_min, k_max, size=n_fs_samples) # randomly generate k values within defined range (random search)
kbest_options = [SelectKBest(f_classif, k=int(k)) for k in k_samples] # f_classif = T-test, loop for different k's, randomized search

# PCA
n_components_samples = rng.randint(2, 20, size=n_fs_samples) # same bounds as for t-test
pca_options = [PCA(n_components=int(n)) for n in n_components_samples] # generating multiple PCA FS options

all_fs_options = kbest_options + pca_options + ['passthrough'] # pass-through makes no extra feature selection, only variance threshold of preprocessing is applied


clfs = [
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    DecisionTreeClassifier(),
    LogisticRegression(max_iter=10000, tol=1e-3) # to allow convergence, max_iter is given
]

param_grids = {
    RandomForestClassifier: {
        'fs': all_fs_options,
        'clf__n_estimators': randint(50, 150),      # Standard=100
        'clf__max_depth': randint(2, 6),            # Standard=None
        'clf__min_samples_split': randint(8, 20),   # Standard=2
        'clf__min_samples_leaf' : randint(3, 10),   # Standard=1
        'clf__max_features': ['sqrt', 'log2']    
    },
    GradientBoostingClassifier: {
        'fs': all_fs_options,
        'clf__n_estimators': randint(100, 250),
        'clf__learning_rate': loguniform(0.001, 0.05),
        'clf__max_depth': randint(2, 6),
        'clf__subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],               #standard=1, can only accept 1 decimal numbers, so this acts as rand(0.5, 1)
        'clf__min_samples_leaf': randint(2, 5)
    },
    DecisionTreeClassifier: {
        'fs': all_fs_options,
        'clf__max_depth': randint(2, 5),             #Standard=None
        'clf__min_samples_split': randint(8, 20),
        'clf__min_samples_leaf': randint(5, 12),        #standard=1
        'clf__criterion': ['gini', 'entropy']
    },  
    LogisticRegression: {
        'fs': all_fs_options,
        'clf__C': loguniform(0.1, 10),             # Standard=1
        'clf__solver': ['saga']
    }
}

#%% Nested Cross-validation Loop

scoring = {
    'accuracy': metrics.make_scorer(metrics.accuracy_score),
    'auc': metrics.make_scorer(metrics.roc_auc_score),
    'f1': metrics.make_scorer(metrics.f1_score),
    'precision': metrics.make_scorer(metrics.precision_score),
    'recall': metrics.make_scorer(metrics.recall_score)
}

results = {}
fitted_searches = {}

for clf in clfs:
    clf_type = type(clf)
    param_grid = param_grids[clf_type]

    pipe = Pipeline([
        ('fs', 'passthrough'), # placeholder, will be overwritten by RandomizedSearchCV
        ('clf', clf)
    ])

    inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42) # 5-fold repeated 3 times (less overfitting), shuffles the order of the data before splitting, with reproducability
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        n_iter=100, # Nr. of times repeated
        cv=inner_cv, 
        scoring='roc_auc', # roc_auc is the most complete metric, which includes both specificity and sensitivity, making it the best for hyperparameter optimalization
        random_state=42,
        error_score=0 # if LogReg produces 0 features, score 0 rather than crash
    )

    cv_results = cross_validate(
        search,
        x_train_final, y=y_train,
        cv=outer_cv,
        scoring=scoring,
        return_train_score=False,   # Not relevant
    )

    results[clf_type.__name__] = cv_results

    search.fit(x_train_final, y_train)
    fitted_searches[clf_type.__name__] = search

    print(clf_type.__name__)
    print(f"  Best inner CV AUC: {search.best_score_:.3f}")
    print(f"  Best params: {search.best_params_}")
    for metric in ['test_accuracy', 'test_auc', 'test_f1', 'test_precision', 'test_recall']:
        mean = cv_results[metric].mean()
        std = cv_results[metric].std()
        print(f"  {metric.replace('test_', '')}: {mean:.3f} ± {std:.3f}")
    print()

#%% Final model training

final_clf = GradientBoostingClassifier() # Best performing model
final_clf_type = type(final_clf)

final_pipe = Pipeline([
    ('fs', 'passthrough'),
    ('clf', final_clf)
])

final_inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

final_search = RandomizedSearchCV(
    estimator=final_pipe,
    param_distributions=param_grids[final_clf_type],
    n_iter=50,
    cv=final_inner_cv,
    scoring='roc_auc',
    refit=True,             # fits best model on full x_train_final after search with best parameters: optimally trained model
    random_state=42,
    error_score=0
)

# Fit on ALL training data (maximize data to fit to, maximalizes performance)
final_search.fit(x_train_final, y_train)
print("Best params:", final_search.best_params_)
print("Final model AUC with best parameters:", final_search.best_score_)

#%% Result of Final Model Training 

final_model = GradientBoostingClassifier(
    learning_rate=np.float64(0.01184431975182039), 
    max_depth=2, 
    min_samples_leaf=4, 
    n_estimators=206, 
    subsample=0.9
)

final_pipe = Pipeline([
        ('fs', 'passthrough'), # placeholder, will be overwritten by RandomizedSearchCV
        ('clf', final_model)
    ])

final_pipe.fit(x_train_final, y_train)
print("Final model AUC:", final_search.best_score_)

#%% Final evaluation on test set (only run once!)

y_pred = final_pipe.predict(x_test_final)
y_prob = final_pipe.predict_proba(x_test_final)[:,1] # Output is 2 columns, this selects only the column "prob_class_1", not class 0

print("Test AUC:      ", metrics.roc_auc_score(y_test, y_prob))
print("Test Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Test F1:       ", metrics.f1_score(y_test, y_pred))
print("Test Precision:", metrics.precision_score(y_test, y_pred))
print("Test Recall:   ", metrics.recall_score(y_test, y_pred))

# %% ROC curve on the test set

y_prob = final_search.predict_proba(x_test_final)[:, 1]

fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_predictions(
    y_test, y_prob,
    name='GradientBoosting (test set)',
    ax=ax
)

# Random classifier diagonal: Representation of random performance (guessing)
ax.plot([0, 1], [0, 1], color='black', linestyle=':', alpha=0.5, label='Random classifier')
ax.set_title('ROC Curve — GradientBoostingClassifier')
ax.legend()
plt.tight_layout()
plt.show()

# %% Learning Curve plot of the final model

train_sizes, train_scores, val_scores = learning_curve(
    final_pipe,
    x_train_final, y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    train_sizes=np.linspace(0.1, 1.0, 10),  # 10 points from 10% to 100% of training data
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(8, 6))

# Training score
ax.plot(train_sizes, train_mean, label='Training AUC', color='blue')
ax.fill_between(train_sizes,
                train_mean - train_std,
                train_mean + train_std,
                alpha=0.15, color='blue')

# Validation score
ax.plot(train_sizes, val_mean, label='Validation AUC', color='green')
ax.fill_between(train_sizes,
                val_mean - val_std,
                val_mean + val_std,
                alpha=0.15, color='green')

ax.set_xlabel('Training set size')
ax.set_ylabel('AUC')
ax.set_title('Learning Curve — GradientBoostingClassifier')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()