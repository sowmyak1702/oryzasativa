from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
    # Define the random forest model
rf_model = RandomForestRegressor()
    # Set the parameter grid for hyperparameter tuning
param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
df=pd.read_csv("data/dataprediction.csv")
x=df["Sequence"]
y=df["Subpopulation"]
le=LabelEncoder()
y2=le.fit_transform(y)
nucleotides = ['A', 'C', 'G', 'T']
one_hot = np.zeros((len(x), len(x[0]) * len(nucleotides)))
for i, seq in enumerate(x):
        for j, nuc in enumerate(seq):
            index = nucleotides.index(nuc)
            one_hot[i, j*len(nucleotides) + index] = 1
y=pd.DataFrame(y2)
X1 = np.concatenate((one_hot, y.values.reshape(-1, 1)), axis=1)
y1=df["Plant Height (cm)"]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=123)
    # Define the randomized search with cross-validation
rf_random = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid, n_iter=50, cv=5, random_state=42, n_jobs=-1)
    # Perform the randomized search with cross-validation on the training data
rf_random.fit(X1_train, y1_train)
    # Define the random forest model with the best hyperparameters
best_rf_model = RandomForestRegressor(**rf_random.best_params_)
    # Perform cross-validation on the training data with the best model
cv_scores = cross_val_score(best_rf_model, X1, y1, cv=5)
    # Fit the best model on the entire dataset
best_rf_model.fit(X1, y1)
# Make predictions on new data
# predictions = best_rf_model.predict(X1_test)
# print('Predictions:', predictions)
# mse = mean_squared_error(y_test, predictions)
# print("Mean Squared Error:", mse)
# df=pd.DataFrame()
# df["Actual"]=y1_test
# df["Predicted"]=predictions
# print(df)
pickle.dump(best_rf_model,open("predheight.pkl","wb"))
