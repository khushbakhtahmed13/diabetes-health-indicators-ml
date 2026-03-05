* Download the **Diabetes Health Indicators Dataset** from Kaggle (or your source) and place the CSV file into the data/raw/ directory.
* Run the preprocessing script to handle missing values and scale features; processed versions will be saved in data/processed/.
* Open main_analysis.ipynb to follow the workflow from data exploration to hyperparameter tuning.
* Regression tuning focuses on finding the optimal alpha for Ridge and Lasso to handle multicollinearity in predictions.
* Classification tuning of Decision Tree Classifier with RandomizedSearchCV to find the best max_depth, min_samples_leaf, and ccp_alpha for diabetes  prediction.
* Evaluate model performance using the specialized functions in src/evaluation.py for confusion matrices, F1-score reporting and performance plots.
* The models/ directory must contain the exported .joblib files of the best binary classification, multi-class classficiation and regression models.


