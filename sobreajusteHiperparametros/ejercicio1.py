from sklearn import datasets
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import math


# Importar conjunto de datos wine
wine = datasets.load_wine()
x = wine.data
y = wine.target
features = wine.feature_names
n_features = len(features)

# Reducir las características a 3
x = x[:, :3]

# Entrenar el regresor MLP con todas las observaciones disponibles
regressor = MLPRegressor(hidden_layer_sizes=(20,) * 100, max_iter=10000)
regressor.fit(x, y)

# Predecir una nueva muestra
new_sample = np.array([[1., 2., 3.]])
predicted_value = regressor.predict(new_sample)

# Validación cruzada de 5 pliegues
kf = KFold(n_splits=5, shuffle=True)
mse_scores = cross_val_score(regressor, x, y, cv=kf, scoring='neg_mean_squared_error')

average_mse = -np.mean(mse_scores)
print('####################################################')
print('Rendimiento con 3 clases:', average_mse)
print(' ')

######################################################################################################################################

# Importar conjunto de datos wine
wine = datasets.load_wine()
x = wine.data
y = wine.target
features = wine.feature_names
n_features = len(features)

# Reducir las características a 3
x = x[:, :3]

# Iterar sobre las capas ocultas
mse_values = []  # Lista para almacenar los valores de MSE

# Iterar sobre las capas ocultas
for n_layers in range(4, 101, 2):
    print('Capas ocultas:', n_layers)

    # Entrenar el regresor MLP con todas las observaciones disponibles
    regressor = MLPRegressor(hidden_layer_sizes=(20,) * n_layers, max_iter=10000)
    regressor.fit(x, y)

    # Predecir una nueva muestra
    new_sample = np.array([[1., 2., 3.]])
    predicted_value = regressor.predict(new_sample)

    # Validación cruzada de 5 pliegues
    kf = KFold(n_splits=5, shuffle=True)
    mse_scores = cross_val_score(regressor, x, y, cv=kf, scoring='neg_mean_squared_error')

    average_mse = -np.mean(mse_scores)
    mse_values.append(average_mse)  # Agregar el valor de MSE a la lista
    print('####################################################')
    print('Rendimiento con 3 clases en ',n_layers,':', average_mse)
    print(' ')

# Graficar los valores de MSE
plt.plot(range(4, 101, 2), mse_values, marker='o')
plt.xlabel('Número de capas ocultas')
plt.ylabel('MSE')
plt.title('Rendimiento del MLP con diferentes capas ocultas')
plt.grid(True)
plt.show()

