
import numpy as np
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

# Training a multi layer percetro for data with two classes
data = np.loadtxt("c:/Users/Admin/Desktop/Semestre 6/deep/ajustesModeloClasificacion/misterious_data_1.txt")
x = data[:,1:]
y = data[:,0]

# Train MLP classifier with all available observations
clf = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=10000)
clf.fit(x, y)

# Applying cross validation
kf = KFold(n_splits=5, shuffle = True)

acc = 0
for train_index, test_index in kf.split(x):
    # Training phase
    x_train = x[train_index, :]
    y_train = y[train_index]

    clf_i = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=10000)
    clf_i.fit(x_train, y_train)

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]    
    y_pred = clf_i.predict(x_test)

    # Calculate confusion matrix and model performance
    cm = confusion_matrix(y_test, y_pred)
    acc_i = (cm[0,0]+cm[1,1])/len(y_test)    



    acc += acc_i 

acc = acc/5
print('ACC = ', acc)

# Training a multi perceptron using data with four classes
data_two = np.loadtxt("c:/Users/Admin/Desktop/Semestre 6/deep/entrenamientoRN/misterious_data_4.txt")
x = data[:,1:]
y = data[:,0]

# Train MLP classifier with all available observations
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=10000)
mlp.fit(x, y)

# Applying cross validation
kf = KFold(n_splits=5, shuffle = True)

acc = 0
for train_index, test_index in kf.split(x):
    # Training phase
    x_train = x[train_index, :]
    y_train = y[train_index]

    mlp_i = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=10000)
    mlp_i.fit(x_train, y_train)

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]    
    y_pred = mlp_i.predict(x_test)

    # Calculate confusion matrix and model performance
    cm = confusion_matrix(y_test, y_pred)
    acc_i = np.trace(cm) / np.sum(cm)   
    acc += acc_i 

acc = acc/5
print('ACC = ', acc)

# Using diabetes dataset to evaluate in a multilayer perceptron

#Loading data and checking it
diabetes = datasets.load_diabetes()
x = diabetes.data
y = diabetes.target
features = diabetes.feature_names
n_features = len(features)

# Entrenando el regresor MLP
regressor = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=10000)
regressor.fit(x, y)

# Inicializando la variable para almacenar el MSE promedio
mse_total = 0

# Aplicando validacion cruzada
kf = KFold(n_splits=5, shuffle=True)

for train_index, test_index in kf.split(x):
    # Dividiendo los datos en conjuntos de entrenamiento y prueba
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Entrenando el regresor en el conjunto de entrenamiento
    regressor.fit(x_train, y_train)

    # Realizando predicciones en el conjunto de prueba
    y_pred = regressor.predict(x_test)

    # Calculando el MSE en el conjunto de prueba
    mse = mean_squared_error(y_test, y_pred)

    # Acumulando el MSE en cada iteración
    mse_total += mse

# Calculando el MSE promedio
mse_avg = mse_total / kf.get_n_splits()

# Imprimiendo el resultado
print("MSE promedio:", mse_avg)

