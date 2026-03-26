# TM10011_PROJECT_Lipoma_Liposarcoma_Group6

This project implements a machine learning model to distinguish lipoma from well-differentiated liposarcoma based on features extracted from T1-weighted MRI. The full pipeline is implemented in final.py.

The data is split into a training and test dataset using a stratified 85/15 split. Preprocessing includes Robust scaling, winsorisation and variance thresholding. Feature selection (SelectKBest, PCA, or none) and classifier choice are optimized using nested cross-validation.

The Gradient Boosting Classifier, the final model, is trained on the full training set and evaluated on the test set. Performance is assessed using metrics such as AUC, accuracy, and F1-score, with confidence intervals obtained through bootstrapping. The script also generates a ROC curve and a learning curve.

# Usage
Install required packages

pip install numpy pandas scikit-learn matplotlib scipy

Run the pipeline:

python final.py

# Authors
Tjitske Pol,
Lars Jongsma,
Fleur Leenheer &
Janneke van Tilburg

# Course
TM10011 Machine Learning
