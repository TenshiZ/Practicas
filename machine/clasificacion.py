import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset from the CSV file
model = pd.read_csv('Breast_Cancer.csv')

# Split the dataset into features (X) and target variable (y)
target = 'Status'
y = model[target]
X = model.drop([target], axis=1)

X = pd.get_dummies(X)

# Encode the target variable with numeric values
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=122)

# Train the XGBoost classifier
my_model = XGBClassifier(n_estimators=1000, learning_rate=0.05, n_jobs=4, early_stopping_rounds=5)
my_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

# Predict the target variable values for the validation set
predict_model = my_model.predict(X_valid)


# Calculate the accuracy
accuracy = accuracy_score(y_valid, predict_model)
precision = precision_score(y_valid, predict_model)

cm = confusion_matrix(y_valid, predict_model)

fig, ax = plt.subplots()

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)

classes = np.unique(y_valid)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks + 0.5, classes)
plt.yticks(tick_marks + 0.5, classes)

plt.xlabel('Etiqueta predicha')
plt.ylabel('Etiqueta real')


plt.title('Matriz de confusión')

plt.savefig('confusion_matrix.png')

print(cm)


f1 = f1_score(y_valid, predict_model)
#predict_model = np.where(predict_model == 0, "Dead", np.where(predict_model == 1, "Alive"))
predict_model = np.where(predict_model == 0, "Dead", "Alive")
y_valid = np.where(y_valid == 0, "Dead", "Alive")
#print(predict_model)
#print(y_valid)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("F1_score:", f1)

