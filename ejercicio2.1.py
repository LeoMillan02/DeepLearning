from sklearn import datasets
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import cross_val_score

#########################################################################################################################################################
# Parte 1
file_path = "c:/Users/Admin/Desktop/Semestre 6/deep/ajustesModeloClasificacion/misterious_data_1.txt"
data = np.genfromtxt(file_path, delimiter='\t')
x = data[:, :-1]
y = data[:, -1]

# Reducir las características a 10
x = x[:, :2]

# Entrenar el regresor MLP con todas las observaciones disponibles
regressor = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=10000)
regressor.fit(x, y)

# Predecir una nueva muestra
new_sample = np.array([[1., 2.]])
predicted_value = regressor.predict(new_sample)
print("Predicción para una nueva observación:", predicted_value)

# Validación cruzada de 5 pliegues
kf = KFold(n_splits=5, shuffle=True)
mse_scores = cross_val_score(regressor, x, y, cv=kf, scoring='neg_mean_squared_error')

average_mse = -np.mean(mse_scores)
print('####################################################')
print('Rendimiento con 2 clases:', average_mse)
print(' ')

#########################################################################################################################################################
#Parte 2

file_path = "c:/Users/Admin/Desktop/Semestre 6/deep/entrenamientoRN/misterious_data_4.txt"
data = np.genfromtxt(file_path, delimiter='\t')
x = data[:, :-1]
y = data[:, -1]

# Reducir las características a 10
x = x[:, :4]

# Entrenar el regresor MLP con todas las observaciones disponibles
regressor = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=10000)  # hidden_layer_sizes controla el número de neuronas de cada capa oculta.
regressor.fit(x, y)

# Predecir una nueva muestra    
new_sample = np.array([[1., 2., 3., 4.]])
predicted_value = regressor.predict(new_sample)
print("Predicción para una nueva observación:", predicted_value)

# Validación cruzada de 5 pliegues
kf = KFold(n_splits=5, shuffle=True)
mse_scores = []

for train_index, test_index in kf.split(x):
    # Fase de entrenamiento
    x_train = x[train_index, :]
    y_train = y[train_index]
    regressor_i = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=10000)
    regressor_i.fit(x_train, y_train)
    
    # Fase de prueba
    x_test = x[test_index, :]
    y_test = y[test_index]
    y_pred = regressor_i.predict(x_test)
    
 # Validación cruzada de 5 pliegues
kf = KFold(n_splits=5, shuffle=True)
mse_scores = cross_val_score(regressor, x, y, cv=kf, scoring='neg_mean_squared_error')

average_mse = -np.mean(mse_scores)
print('####################################################')
print('Rendimiento con 4 clases:', average_mse)
print(' ')

#########################################################################################################################################################
#Parte 3

# Importar conjunto de datos diabetes
diabetes = datasets.load_diabetes()
x = diabetes.data
y = diabetes.target
features = diabetes.feature_names
n_features = len(features)


# Reducir las características a 4
x = x[:, :4]

# Entrenar el regresor MLP con todas las observaciones disponibles
regressor = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=10000)  # hidden_layer_sizes controla el número de neuronas de cada capa oculta.
regressor.fit(x, y)

# Predecir una nueva muestra    
new_sample = np.array([[1., 2., 3., 4.]])
predicted_value = regressor.predict(new_sample)
print("Predicción para una nueva observación:", predicted_value)

# Validación cruzada de 5 pliegues
kf = KFold(n_splits=5, shuffle=True)
mse_scores = []

for train_index, test_index in kf.split(x):
    # Fase de entrenamiento
    x_train = x[train_index, :]
    y_train = y[train_index]
    regressor_i = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=10000)
    regressor_i.fit(x_train, y_train)
    
    # Fase de prueba
    x_test = x[test_index, :]
    y_test = y[test_index]
    y_pred = regressor_i.predict(x_test)
    
    # Calcular el error cuadrático medio
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

average_mse = np.mean(mse_scores)
print('####################################################')
print('Error cuadrático medio promedio (MSE):', average_mse)
print(' ')
