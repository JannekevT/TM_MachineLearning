#%%
from worclipo.load_data import load_data
from sklearn.model_selection import train_test_split

# General packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets as ds
import seaborn
from sklearn import metrics
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

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
#%%
data = load_data()
x = data.drop(["label"], axis='columns').values
y = data["label"].map({'lipoma':0, 'liposarcoma':1})
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42, stratify=y) # Stratified test split, random state for reproducability, 85/15

#%%
x_fs = data.drop(["label"], axis='columns').values
y_fs = data["label"].map({'lipoma':0, 'liposarcoma':1})

selector_var = VarianceThreshold(threshold=0.1)
x_fs_high_var = selector_var.fit_transform(x_fs)
#print(f"Features after variance filtering: {x_fs_high_var.shape[1]}")

k = int(0.2 * x_fs_high_var.shape[1])
selector_kbest = SelectKBest(f_classif, k=k)
x_selected = selector_kbest.fit_transform(x_fs_high_var, y)

x_fs_train, x_fs_test, y_fs_train, y_fs_test = train_test_split(x_selected, y_fs, test_size=0.15, random_state=42, stratify=y_fs) # Stratified test split, random state for reproducability, 85/15

#%%
clsfs = [
    LinearDiscriminantAnalysis(solver="eigen", shrinkage=True),
    QuadraticDiscriminantAnalysis(solver="eigen", shrinkage=True),
    GaussianNB(),
    LogisticRegression(),
    SGDClassifier(),
    SVC(),
    NuSVC(),
    DecisionTreeClassifier(),
    KNeighborsClassifier(),
    RandomForestClassifier(), 
    GradientBoostingClassifier(), 
    AdaBoostClassifier()
    ]

clfs_fit = list()

#%%
scoring = {
    'accuracy': metrics.make_scorer(metrics.accuracy_score),
    'auc': metrics.make_scorer(metrics.roc_auc_score),
    'f1': metrics.make_scorer(metrics.f1_score),
    'precision': metrics.make_scorer(metrics.precision_score),
    'recall': metrics.make_scorer(metrics.recall_score)
}

# %% Without Feature Selection
for clf in clsfs:
    cv = StratifiedKFold(n_splits=5)
    
    cv_results = cross_validate(
        clf, x_train, y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    print(type(clf).__name__)
    for metric in ['test_accuracy', 'test_auc', 'test_f1', 'test_precision', 'test_recall']:
        mean = cv_results[metric].mean()
        std = cv_results[metric].std()
        print(f"{metric.split('test_'[1].title())}: {mean:.3f} +- {std:.3f}")
    print(" ")

# %% With Feature Selection
for clf in clsfs:
    cv = StratifiedKFold(n_splits=5)
    
    cv_results = cross_validate(
        clf, x_fs_train, y_fs_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    print(type(clf).__name__)
    for metric in ['test_accuracy', 'test_auc', 'test_f1', 'test_precision', 'test_recall']:
        mean = cv_results[metric].mean()
        std = cv_results[metric].std()
        print(f"{metric.split('test_'[1].title())}: {mean:.3f} +- {std:.3f}")
    print(" ")
# %% Nested Cross-validation loop

parameters_grid = {}
N_trials = 20
nested_scores = np.zeros(N_trials)

for clf in clsfs:
    for i in range(N_trials):
        inner_cv = StratifiedKFold(n_splits=5)
        outer_cv = StratifiedKFold(n_splits=5)

        classifier = GridSearchCV(estimator=clf, param_grid=parameters_grid, cv=inner_cv)
        nested_scores = cross_validate(
            classifier, x_fs_train, y_fs_train,
            cv=outer_cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )

        print(type(clf).__name__)
        for metric in ['test_accuracy', 'test_auc', 'test_f1', 'test_precision', 'test_recall']:
            mean = nested_scores[metric].mean()
            std = nested_scores[metric].std()
            print(f"{metric.split('test_'[1].title())}: {mean:.3f} +- {std:.3f}")
        print(" ")