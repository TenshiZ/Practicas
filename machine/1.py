import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

label_encoder = LabelEncoder()

ruta = "iris.csv"
model = pd.read_csv(ruta)

columnas = ["sepal.length", "sepal.width", "petal.length", "petal.width"]

s = (model.dtypes == 'object')
object_cols = list(s[s].index)
print(object_cols)

ordinal_encoder = OrdinalEncoder()

model_copy = model.copy()
model_copy[object_cols] = ordinal_encoder.fit_transform(model[object_cols])

y = model_copy.variety
X = model_copy[columnas]

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=1)

forest = RandomForestRegressor(random_state=0)
tree = DecisionTreeRegressor(random_state=0)
tree.fit(train_X, train_y)
forest.fit(train_X, train_y)

f_predictions = forest.predict(val_X)
predictions = tree.predict(val_X)

f_error = mean_absolute_error(val_y, f_predictions)
error = mean_absolute_error(val_y, predictions)

# Reshape predictions to a 2D array
predictions_reshaped = predictions.reshape(-1, 1)
f_predictions_reshaped = f_predictions.reshape(-1, 1)
# Inverse transform the reshaped predictions
predictions_inverse = ordinal_encoder.inverse_transform(predictions_reshaped)
f_predictions_inverse = ordinal_encoder.inverse_transform(f_predictions_reshaped)

# Reshape val_y to a 2D array
val_y_reshaped = val_y.values.reshape(-1, 1)

# Inverse transform the reshaped val_y
val_y_inverse = ordinal_encoder.inverse_transform(val_y_reshaped)
print("Datos")
print(val_X)
print("Resultados")
print(val_y_inverse)
print("Predicciones")
print(predictions_inverse.flatten())
print('Error: ' + str(round((error * 100), 2)) + '%')

print("Predicciones")
print(f_predictions_inverse.flatten())
print('Error: ' + str(round((f_error * 100), 2)) + '%')
