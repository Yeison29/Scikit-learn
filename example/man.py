# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd
from tabulate import tabulate

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.ticker as ticker
import seaborn as sns
import statsmodels.api as sm

# Preprocesado y modelado
# ==============================================================================
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence
import optuna

# Varios
# ==============================================================================
import multiprocessing
import random
from itertools import product
from fitter import Fitter, get_common_distributions

datos = pd.read_csv('../SaratogaHouses.csv')

datos.columns = ["precio", "tamaño_terreno", "antiguedad", "precio_terreno", "metros_habitables",
                 "universitarios", "dormitorios", "chimenea", "banyos", "habitaciones",
                 "calefaccion", "consumo_calefacion", "desague", "vistas_lago", "nueva_construccion",
                 "aire_acondicionado"]

datos.info()

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 6))
sns.kdeplot(
    datos.precio,
    fill=True,
    color="blue",
    ax=axes[0]
)
sns.rugplot(
    datos.precio,
    color="blue",
    ax=axes[0]
)
axes[0].set_title("Distribución original", fontsize='medium')
axes[0].set_xlabel('precio', fontsize='small')
axes[0].tick_params(labelsize=6)

sns.kdeplot(
    np.sqrt(datos.precio),
    fill=True,
    color="blue",
    ax=axes[1]
)
sns.rugplot(
    np.sqrt(datos.precio),
    color="blue",
    ax=axes[1]
)
axes[1].set_title("Transformación raíz cuadrada", fontsize='medium')
axes[1].set_xlabel('sqrt(precio)', fontsize='small')
axes[1].tick_params(labelsize=6)

sns.kdeplot(
    np.log(datos.precio),
    fill=True,
    color="blue",
    ax=axes[2]
)
sns.rugplot(
    np.log(datos.precio),
    color="blue",
    ax=axes[2]
)
axes[2].set_title("Transformación logarítmica", fontsize='medium')
axes[2].set_xlabel('log(precio)', fontsize='small')
axes[2].tick_params(labelsize=6)

fig.tight_layout()

plt.show()


# Variables numéricas
# ==============================================================================
print(datos.select_dtypes(include=['float64', 'int']).describe())

#distribuciones
distribuciones = ['cauchy', 'chi2', 'expon', 'exponpow', 'gamma',
                  'norm', 'powerlaw', 'beta', 'logistic']

# Fitterb nos ayuda a mirar que distribución se ajusta mejor a l conjunto de datos
fitter = Fitter(datos.precio, distributions=distribuciones)
fitter.fit()
fitter.summary(Nbest=10, plot=True)

print(fitter.get_best(method='sumsquare_error'))
print(fitter.summary())

# Gráfico de distribución para cada variable numérica
# ==============================================================================
# Ajustar número de subplots en función del número de columnas
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 5))
axes = axes.flat
columnas_numeric = datos.select_dtypes(include=['float64', 'int']).columns
columnas_numeric = columnas_numeric.drop('precio')

for i, colum in enumerate(columnas_numeric):
    sns.histplot(
        data=datos,
        x=colum,
        stat="count",
        kde=True,
        color=(list(plt.rcParams['axes.prop_cycle']) * 2)[i]["color"],
        line_kws={'linewidth': 2},
        alpha=0.3,
        ax=axes[i]
    )
    axes[i].set_title(colum, fontsize=7, fontweight="bold")
    axes[i].tick_params(labelsize=6)
    axes[i].set_xlabel("")

fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Distribución variables numéricas', fontsize=10, fontweight="bold")

plt.show()

# Se convierte la variable chimenea tipo string
# ==============================================================================
datos.chimenea = datos.chimenea.astype("str")

# Gráfico de distribución para cada variable numérica
# ==============================================================================
# Ajustar número de subplots en función del número de columnas
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 5))
axes = axes.flat
print(datos.select_dtypes(include=['float64', 'int']))
columnas_numeric = datos.select_dtypes(include=['float64', 'int']).columns
columnas_numeric = columnas_numeric.drop('precio')

print(columnas_numeric)

for i, colum in enumerate(columnas_numeric):
    sns.regplot(
        x=datos[colum],
        y=datos['precio'],
        color="gray",
        marker='.',
        scatter_kws={"alpha": 0.4},
        line_kws={"color": "r", "alpha": 0.7},
        ax=axes[i]
    )
    axes[i].set_title(f"precio vs {colum}", fontsize=7, fontweight="bold")
    # axes[i].ticklabel_format(style='sci', scilimits=(-4,4), axis='both')
    axes[i].yaxis.set_major_formatter(ticker.EngFormatter())
    axes[i].xaxis.set_major_formatter(ticker.EngFormatter())
    axes[i].tick_params(labelsize=6)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

# Se eliminan los axes vacíos
for i in [8]:
    fig.delaxes(axes[i])

fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Correlación con precio', fontsize=10, fontweight="bold")

plt.show()


# Correlación entre columnas numéricas
# ==============================================================================
def tidy_corr_matrix(corr_mat):
    '''
    Función para convertir una matrix de correlación de pandas en formato tidy
    '''
    corr_mat = corr_mat.stack().reset_index()
    corr_mat.columns = ['variable_1', 'variable_2', 'r']
    corr_mat = corr_mat.loc[corr_mat['variable_1'] != corr_mat['variable_2'], :]
    corr_mat['abs_r'] = np.abs(corr_mat['r'])
    corr_mat = corr_mat.sort_values('abs_r', ascending=False)

    return corr_mat


corr_matrix = datos.select_dtypes(include=['float64', 'int']).corr(method='pearson')
print(tidy_corr_matrix(corr_matrix))

# Heatmap matriz de correlaciones
# ==============================================================================
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

sns.heatmap(
    corr_matrix,
    annot=True,
    cbar=True,
    annot_kws={"size": 6},
    vmin=-1,
    vmax=1,
    center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    ax=ax
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right',
)

ax.tick_params(labelsize=8)
plt.subplots_adjust(top=0.9, left=0.2)
fig.suptitle('Correlaciones', fontsize=10, fontweight="bold")

plt.show()
# Variables cualitativas (tipo object)
# ==============================================================================
print('Variables cualitativas')
print(datos.select_dtypes(include=['object']).describe())

# Gráfico para cada variable cualitativa
# ==============================================================================
# Ajustar número de subplots en función del número de columnas
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 5))
axes = axes.flat
columnas_object = datos.select_dtypes(include=['object']).columns

for i, colum in enumerate(columnas_object):
    sns.barplot(
        datos[colum].value_counts(),
        color=(list(plt.rcParams['axes.prop_cycle']) * 2)[i]["color"],
        ax=axes[i]
    )

    print(datos[colum].value_counts())
    axes[i].set_title(colum, fontsize=7, fontweight="bold")
    axes[i].tick_params(labelsize=6)
    axes[i].set_xlabel("")

# Se eliminan los axes vacíos
for i in [7, 8]:
    fig.delaxes(axes[i])

fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Distribución variables cualitativas',
             fontsize=10, fontweight="bold")
plt.show()

#puedo juntar las columnas
dic_replace = {'2': "2_mas",
               '3': "2_mas",
               '4': "2_mas"}

datos['chimenea'] = (
    datos['chimenea']
    .map(dic_replace)
    .fillna(datos['chimenea'])
)

# Gráfico relación entre el precio y cada cada variables cualitativas
# ==============================================================================
# Ajustar número de subplots en función del número de columnas
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 5))
axes = axes.flat
columnas_object = datos.select_dtypes(include=['object']).columns

for i, colum in enumerate(columnas_object):
    sns.violinplot(
        x=colum,
        y='precio',
        data=datos,
        color=(list(plt.rcParams['axes.prop_cycle']) * 2)[i]["color"],
        ax=axes[i]
    )
    axes[i].set_title(f"precio vs {colum}", fontsize=7, fontweight="bold")
    axes[i].yaxis.set_major_formatter(ticker.EngFormatter())
    axes[i].tick_params(labelsize=6)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

# Se eliminan los axes vacíos
for i in [7, 8]:
    fig.delaxes(axes[i])

fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Distribución del precio por grupo', fontsize=10, fontweight="bold")

plt.show()

