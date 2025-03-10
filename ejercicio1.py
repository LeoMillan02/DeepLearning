import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score

#########################################################################################################################################################

# Funcion del perceptron
def perceptron(x, w):
    ws = sum(x * w)
    yt = 0
    if ws < 0:
        yt = -1
    elif ws > 0:
        yt = 1

    return yt

def perceptron_mult(x, w):
    yp = []
    for xi in x:
        yp.append(perceptron(xi, w))

    return np.array(yp)

##############################################################################################################################################

# Gradiente estoica descendiente
def train_perceptron_sg(x, y, x_test = None, y_test = None, n_epochs = 20, alpha = 0.001):
    n = x.shape[0]
    n_features = x.shape[1]

    #Inicializacion aleatoria
    w = 2 * np.random.rand(n_features) - 1

    # Calcular modelo
    epoch_acc = []
    for epoch in range(n_epochs):
        # Indices aleatorios
        indices = np.random.permutation(n)
        for index in indices:
            # Evaluar perceptron
            yp = perceptron(x[index, :], w)

            # Actualizar pesos
            w = w + alpha * (y[index] - yp) * x[index, :]
    
        # Evaluar rendimiento
        if x_test is not None and y_test is not None:
            y_test_pred = perceptron_mult(x_test, w)
            acc = accuracy_score(y_test, y_test_pred)
            epoch_acc.append(acc)

    # Regresar modelo y promedio
    if x_test is not None and y_test is not None:
        epoch_acc = np.array(epoch_acc)
        return {'w':w, 'acc':epoch_acc[-1], 'epoch_acc': epoch_acc}
    
    return {'w':w}

##############################################################################################################################################

#Gradiente de lote
def train_perceptron_batch(x, y, x_test = None, y_test = None, n_epochs = 20, alpha = 0.001):
    n = x.shape[0]
    n_features = x.shape[1]

   #Inicializacion aleatoria
    w = 2 * np.random.rand(n_features) - 1

    # Calcular modelo
    epoch_acc = []
    for epoch in range(n_epochs):
        grad = np.zeros(n_features)

        # Calcular modelo
        for i in range(n):
           # Evaluar perceptron
            yp = perceptron(x[i,:], w)

            # Calcular gradiente
            grad += (y[i] - yp) * x[i,:]

        # Actualizar pesos
        w = w + alpha * grad

        #Evaluar rendimiento
        if x_test is not None and y_test is not None:
            y_test_pred = perceptron_mult(x_test, w)
            acc = accuracy_score(y_test, y_test_pred)
            epoch_acc.append(acc)
    
    # Regresar modelo y promedio
    if x_test is not None and y_test is not None:
        epoch_acc = np.array(epoch_acc)
        return {'w':w, 'acc':epoch_acc[-1], 'epoch_acc':epoch_acc}
    
    return {'w':w}
#########################################################################################################################################################

# Optimización de mini lote 
def train_perceptron_mini_batch(x, y, x_test = None, y_test = None, n_epochs = 20, alpha = 0.001, batch_size = 5):
    n = x.shape[0]
    n_features = x.shape[1]

  # Inicialización aleatoria
    w = 2 * np.random.rand(n_features) - 1

    # Calcular modelo
    n_updates = n//batch_size + int(n%batch_size > 0)
    epoch_acc = []
    for epoch in range(n_epochs):
        # Indices aleatorios
        indices = np.random.permutation(n)

        # Actualizar modelos
        j = 0
        for i in range(n_updates):
            #Inicializar gradiente
            grad = np.zeros(n_features)

            # Calcular gradiente
            counter = 0
            for p in range(batch_size):
                if j >= n:
                    continue

                # Evaluar perceptron
                yp = perceptron(x[indices[j], :], w)

                # Calcular gradiente
                grad += (y[indices[j]] - yp) * x[indices[j], :]

                # Actuazlizar contadores
                counter += 1
                j += 1

            # Actualizar pesos
            w = w + alpha * grad

        # Evaluar rendimiento
        if x_test is not None and y_test is not None:
            y_test_pred = perceptron_mult(x_test, w)
            acc = accuracy_score(y_test, y_test_pred)
            epoch_acc.append(acc)

    if x_test is not None and y_test is not None:
        epoch_acc = np.array(epoch_acc)
        return {'w':w, 'acc':epoch_acc[-1], 'epoch_acc':epoch_acc}
    
    return {'w':w}
#########################################################################################################################################################

# Cargas datos
data = np.loadtxt("c:/Users/Admin/Desktop/Semestre 6/deep/ajustesModeloClasificacion/misterious_data_1.txt")
x = data[:,1:]
y = data[:,0]

n = x.shape[0]
nf = x.shape[1]

# Preparar datos
x = np.hstack((x, np.ones((n, 1))))
y[y==1] = -1
y[y==2] = 1

n_epochs = 100
alpha = 0.001

#########################################################################################################################################################

# Evaluar gradiente estoica usando validacion cruzada
nk = 10
kf = StratifiedKFold(n_splits=nk, shuffle=True)

acc = 0
epoch_acc = np.zeros(n_epochs)

for train_index, test_index in kf.split(x,y):
    x_train = x[train_index, :]
    y_train = y[train_index]

    x_test = x[test_index, :]
    y_test = y[test_index]

    # Entrenar perceptron
    model = train_perceptron_sg(x_train, y_train, x_test = x_test, y_test= y_test, n_epochs= n_epochs, alpha= alpha)
    acc += model['acc']
    epoch_acc += model['epoch_acc']

acc /= nk
epoch_acc /= nk
print('Error promedio de gradiente estoica: ', acc)

plt.plot(np.arange(1, n_epochs + 1, 1), epoch_acc)
plt.title('Error promedio vs Epoca:Optimización de Gradiente estoica')
plt.xlabel('Epoca')
plt.ylabel('Error')
plt.show()

#########################################################################################################################################################

# Evaluar usando optimizacion de lote
nk = 10
kf = StratifiedKFold(n_splits=nk, shuffle=True)

acc = 0
epoch_acc = np.zeros(n_epochs)

for train_index, test_index in kf.split(x,y):
    x_train = x[train_index, :]
    y_train = y[train_index]

    x_test = x[test_index, :]
    y_test = y[test_index]

    # Entrenar perceptron
    model = train_perceptron_batch(x_train, y_train, x_test = x_test, y_test= y_test, n_epochs= n_epochs, alpha= alpha)
    acc += model['acc']
    epoch_acc += model['epoch_acc']

acc /= nk
epoch_acc /= nk
print('Error promedio optimización de lote: ', acc)

plt.plot(np.arange(1, n_epochs + 1, 1), epoch_acc)
plt.title('Error promedio vs Epoca:Optimización de lote')
plt.xlabel('Epoca')
plt.ylabel('Error')
plt.show()

#########################################################################################################################################################

# Evaluar usando optimizacion de mini lote
nk = 10
kf = StratifiedKFold(n_splits=nk, shuffle=True)

acc = 0
epoch_acc = np.zeros(n_epochs)

for train_index, test_index in kf.split(x,y):
    x_train = x[train_index, :]
    y_train = y[train_index]

    x_test = x[test_index, :]
    y_test = y[test_index]

    # Train perceptron
    model = train_perceptron_mini_batch(x_train, y_train, x_test = x_test, y_test= y_test, n_epochs= n_epochs, alpha= alpha)
    acc += model['acc']
    epoch_acc += model['epoch_acc']

acc /= nk
epoch_acc /= nk
print('Error promedio optimización de mini lote: ', acc)

plt.plot(np.arange(1, n_epochs + 1, 1), epoch_acc)
plt.title('Error promedio vs Epoca:Optimización de mini lote ')
plt.xlabel('Epoca')
plt.ylabel('Error')
plt.show()

############################################################################################################################################
#EJERCICIO 2

# Utilizar clasificador SVM
data_emoji = np.loadtxt("c:/Users/Admin/Desktop/Semestre 6/deep/ajustesModeloClasificacion/misterious_data_1.txt")
x = data[:,1:]
y = data[:,0]

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle = True)

cv_y_test = []
cv_y_pred = []

# Evaluar clasificador SVM 
for train_index, test_index in kf.split(x):
    
    # Entrenar fase
    x_train = x[train_index, :]
    y_train = y[train_index]

    clf_cv = SVC(kernel = 'linear')
    clf_cv.fit(x_train, y_train)

    # Probar fase
    x_test = x[test_index, :]
    y_test = y[test_index]    

    y_pred = clf_cv.predict(x_test)
    
    cv_y_test.append(y_test)
    cv_y_pred.append(y_pred)
    
#########################################################################################################################################################   
    
# Utilizar gradiente estoica
n_epochs = 100
alpha = 0.001

model = train_perceptron_sg(x, y, x_test = x_test, y_test = y_test, n_epochs = n_epochs, alpha = alpha)

plt.plot(np.arange(1, n_epochs + 1, 1), model['epoch_acc'])
plt.title('Error promedio vs Epoca: Gradiente estocastica SVM')
plt.xlabel('Epoca')
plt.ylabel('Error')
plt.show()

#########################################################################################################################################################

# Utilizar optimizacion de lote
n_epochs = 100
alpha = 0.001

model = train_perceptron_batch(x, y, x_test = x_test, y_test = y_test, n_epochs = n_epochs, alpha = alpha)

plt.plot(np.arange(1, n_epochs + 1, 1), model['epoch_acc'])
plt.title('Error promedio vs Epoca: Optimización de lote SVM')
plt.xlabel('Epoca')
plt.ylabel('Error')
plt.show()

#########################################################################################################################################################

# Utilizar optimizacion de mini lote 
n_epochs = 100
alpha = 0.001

model = train_perceptron_mini_batch(x, y, x_test = x_test, y_test = y_test, n_epochs = n_epochs, alpha = alpha)

plt.plot(np.arange(1, n_epochs + 1, 1), model['epoch_acc'])
plt.title('Error promedio vs Epoca:Optimización de mini lote SVM')
plt.xlabel('Epoca')
plt.ylabel('Error')
plt.show()
