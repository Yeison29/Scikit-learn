import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

datos = pd.read_csv('../data.csv')
print(datos.info())
print('------------------------------------')
print(datos['YEAR/MES'])
datos['MES'] = datos['YEAR/MES'].apply(lambda x: x[-3:])
datos = datos.sort_index(ascending=False).reset_index().drop('index', axis=1)
datos_train = datos
print(datos.head(12))


X_train, X_test, y_train, y_test = train_test_split(
    datos.drop('PRECIO', axis='columns'),
    datos['PRECIO'],
    train_size=0.98,
    shuffle=False
)
print('----------')
print(X_train, y_train)
print('----------')
datos_train['index'] = datos_train.index
print(X_test, y_test)

# datos.columns = ["precio", "tamaño_terreno", "antiguedad", "precio_terreno", "metros_habitables",
#                  "universitarios", "dormitorios", "chimenea", "banyos", "habitaciones",
#                  "calefaccion", "consumo_calefacion", "desague", "vistas_lago", "nueva_construccion",
#                  "aire_acondicionado"]
#
# print(datos['precio'])
#
model = make_pipeline(PolynomialFeatures(degree=7),
                      LinearRegression(fit_intercept=False))
model.fit(X_train.index.values.reshape(-1, 1), y_train.values.reshape(-1, 1))


y_test = model.predict(datos_train['index'].values.reshape(-1, 1))


def plot_prediction(x, y, x_test, y_test, model, original_x, original_y):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True, tight_layout=True)
    ax[0].scatter(x, y, s=50, label='Datos-Entrenamiento', alpha=0.5)
    ax[0].scatter(x_test, y_test, s=50, label='Datos-Predichos', alpha=0.05, color='purple')
    ax[0].scatter(original_x, original_y, s=50, label='Datos-originales', alpha=0.05, color='orange')
    ax[0].plot(x_test, y_test, label='Linea-predicha', lw=1, color='purple')
    ax[0].legend()
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[1].scatter(original_y, original_y - model.predict(original_x.reshape(-1, 1)), alpha=0.5)
    ax[1].axhline(0, c='k', ls='--', alpha=0.5)
    ax[1].set_xlabel('y')
    ax[1].set_ylabel('residuos')


plot_prediction(X_train.index.values.reshape(-1, 1), y_train.values.reshape(-1, 1),
                datos_train.index.values.reshape(-1, 1), y_test, model,
                datos.index.values.reshape(-1, 1), datos['PRECIO'].values.reshape(-1, 1))
plt.show()

datos_train['PREDICION'] = y_test.round(4)

print(datos_train)

#---Red-Neuronal---


