import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

# Parte 3

file_path = "c:/Users/Admin/Desktop/Semestre 6/deep/sobreajusteHiperparametros/misterious_data_5.txt"
data = np.loadtxt(file_path)
x = data[:, :-1]
y = data[:, -1]

# Reducir las características a 4
x = x[:, :4]

# Rango de capas ocultas
num_layers_range = range(10, 101, 10)

# Validación cruzada de 5 pliegues para cada número de capas ocultas
mse_scores = []

for num_layers in num_layers_range:
    kf = KFold(n_splits=5, shuffle=True)
    mse_scores_i = []

    for train_index, test_index in kf.split(x):
        # Fase de entrenamiento
        x_train = x[train_index, :]
        y_train = y[train_index]
        regressor_i = MLPRegressor(hidden_layer_sizes=(num_layers,), max_iter=10000)
        regressor_i.fit(x_train, y_train)

        # Fase de prueba
        x_test = x[test_index, :]
        y_test = y[test_index]
        y_pred = regressor_i.predict(x_test)

        # Calcular el error cuadrático medio
        mse = mean_squared_error(y_test, y_pred)
        mse_scores_i.append(mse)

    average_mse_i = np.mean(mse_scores_i)
    mse_scores.append(average_mse_i)

# Graficar los resultados punto a punto
plt.scatter(num_layers_range, mse_scores)
plt.xlabel('Número de capas ocultas')
plt.ylabel('Error cuadrático medio promedio (MSE)')
plt.title('Evaluación del número de capas ocultas')
plt.show()

