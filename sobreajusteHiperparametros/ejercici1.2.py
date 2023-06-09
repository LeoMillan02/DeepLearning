import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn import datasets

# Importar conjunto de datos wine
wine = datasets.load_wine()
x = wine.data
y = wine.target
features = wine.feature_names
n_features = len(features)

# Reducir las características a 3
x = x[:, :3]

# Definir los valores del término de regularización
regularization_values = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 15, 25, 30, 35, 45, 50, 60, 70, 80, 90, 100]

for regularization_value in regularization_values:
    # Entrenar el regresor MLP con todas las observaciones disponibles
    regressor = MLPRegressor(hidden_layer_sizes=(20,) * 5, max_iter=10000, alpha=regularization_value)
    regressor.fit(x, y)

    # Validación cruzada de 5 pliegues
    kf = KFold(n_splits=5, shuffle=True)
    mse_scores = cross_val_score(regressor, x, y, cv=kf, scoring='neg_mean_squared_error')

    average_mse = -np.mean(mse_scores)
    print('####################################################')
    print('Rendimiento con término de regularización =', regularization_value)
    print('MSE promedio:', average_mse)
    print(' ')

