import pandas as pd


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

url = "Breast_Cancer.csv"
model = pd.read_csv(url)
print(model.head())
y = model['Tumor Size']
X = model.drop(['Tumor Size'], axis = 1)

train_X_full, val_X_full, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=1)

categorical_cols = [cname for cname in train_X_full.columns if train_X_full[cname].nunique() < 10 and train_X_full[cname].dtype == "object"]
numerical_cols = [cname for cname in train_X_full.columns if train_X_full[cname].dtype in ['int64', 'float64']]

my_cols = categorical_cols + numerical_cols
train_X = train_X_full[my_cols].copy()
val_X = val_X_full[my_cols].copy()


numerical_transformer = SimpleImputer(strategy='constant')

categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer,numerical_cols), ('cat', categorical_transformer, categorical_cols)])

my_model = XGBRegressor(n_estimator = 1000, learning_rate = 0.05 , n_jobs = 4)

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),('model', my_model)])
my_pipeline .fit(train_X, train_y)#, early_stopping_rounds = 5, eval_set=[(val_X, val_y)], verbose = False

predict_model = my_pipeline.predict(val_X)

error = mean_absolute_error(val_y, predict_model)
print(error)
