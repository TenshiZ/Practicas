import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


model = pd.read_csv('Breast_Cancer.csv')


target = 'Tumor Size' #regresion ya que no buscamos clasificarlos solo su numero aprox
y = model[target]            
X =model.drop([target], axis=1)


X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, test_size=0.2,random_state=0)

X_train = pd.get_dummies(X_train_full)
X_valid = pd.get_dummies(X_valid_full)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

my_model = XGBRegressor(n_estimators = 1000, learning_rate = 0.05 , n_jobs = 4, early_stopping_rounds = 5)

my_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose = False) 

predict_model = my_model.predict(X_valid)



print(predict_model)
print(y_valid)

mae = mean_absolute_error(y_valid, predict_model)
mse = mean_squared_error(y_valid, predict_model)
r2 = r2_score(y_valid, predict_model)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)
