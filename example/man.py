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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector
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
print(corr_matrix)

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

X_train, X_test, y_train, y_test = train_test_split(
    datos.drop('precio', axis='columns'),
    datos['precio'],
    train_size=0.8,
    random_state=1234,
    shuffle=True
)

print("Partición de entrenamento")
print("-----------------------")
print(y_train.describe())

print("Partición de test")
print("-----------------------")
print(y_test.describe())

# Se estandarizan las columnas numéricas y se hace one-hot-encoding de las
# columnas cualitativas. Para mantener las columnas a las que no se les aplica
# ninguna transformación se tiene que indicar remainder='passthrough'.
numeric_cols = X_train.select_dtypes(include=['float64', 'int']).columns.to_list()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.to_list()

preprocessor = ColumnTransformer(
    [('scale', StandardScaler(), numeric_cols),
     ('onehot', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
    remainder='passthrough')

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# Convertir el output en dataframe y añadir el nombre de las columnas
# ==============================================================================
encoded_cat = preprocessor.named_transformers_['onehot'].get_feature_names_out(cat_cols)
nombre_columnas = np.concatenate([numeric_cols, encoded_cat])
X_train_prep = preprocessor.transform(X_train)
X_train_prep = pd.DataFrame(X_train_prep, columns=nombre_columnas)
X_train_prep.head(3)

preprocessor = ColumnTransformer(
    [('scale', StandardScaler(), numeric_cols),
     ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)],
    remainder='passthrough',
    verbose_feature_names_out=False
).set_output(transform="pandas")

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

print(X_train_prep.head(3))

numeric_cols = X_train.select_dtypes(include=['float64', 'int']).columns.to_list()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.to_list()

# Transformaciones para las variables numéricas
numeric_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

# Transformaciones para las variables categóricas
categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, cat_cols)
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
).set_output(transform="pandas")

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)
print(X_train_prep.head(3))

from sklearn import set_config

set_config(display='diagram')

print(preprocessor)

set_config(display='text')

from sklearn.linear_model import Ridge

# Preprocedado
# ==============================================================================

# Identificación de columnas numéricas y categóricas
numeric_cols = X_train.select_dtypes(include=['float64', 'int']).columns.to_list()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.to_list()

# Transformaciones para las variables numéricas
numeric_transformer = Pipeline(
    steps=[('scaler', StandardScaler())]
)

# Transformaciones para las variables categóricas
categorical_transformer = Pipeline(
    steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, cat_cols)
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
).set_output(transform="pandas")

# Pipeline
# ==============================================================================

# Se combinan los pasos de preprocesado y el modelo en un mismo pipeline
pipe = Pipeline([('preprocessing', preprocessor),
                 ('modelo', Ridge())])

# Train
# ==============================================================================
# Se asigna el resultado a _ para que no se imprima por pantalla
_ = pipe.fit(X=X_train, y=y_train)

# Validación cruzada
# ==============================================================================
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    estimator=pipe,
    X=X_train,
    y=y_train,
    scoring='neg_root_mean_squared_error',
    cv=5
)

print(f"Métricas validación cruzada: {cv_scores}")
print(f"Média métricas de validación cruzada: {cv_scores.mean()}")

# Validación cruzada repetida
# ==============================================================================
from sklearn.model_selection import RepeatedKFold

cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=123)
cv_scores = cross_val_score(
    estimator=pipe,
    X=X_train,
    y=y_train,
    scoring='neg_root_mean_squared_error',
    cv=cv
)

print(f"Métricas de validación cruzada: {cv_scores}")
print("")
print(f"Média métricas de validación cruzada: {cv_scores.mean()}")

# Validación cruzada repetida con múltiples métricas
# ==============================================================================
from sklearn.model_selection import cross_validate

cv = RepeatedKFold(n_splits=3, n_repeats=5, random_state=123)
cv_scores = cross_validate(
    estimator=pipe,
    X=X_train,
    y=y_train,
    scoring=('r2', 'neg_root_mean_squared_error'),
    cv=cv,
    return_train_score=True
)

# Se convierte el diccionario a dataframe para facilitar la visualización
cv_scores = pd.DataFrame(cv_scores)
print(cv_scores)

# Distribución del error de validación cruzada
# ==============================================================================
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 5), sharex=True)

sns.kdeplot(
    cv_scores['test_neg_root_mean_squared_error'],
    fill=True,
    alpha=0.3,
    color="firebrick",
    ax=axes[0]
)
sns.rugplot(
    cv_scores['test_neg_root_mean_squared_error'],
    color="firebrick",
    ax=axes[0]
)
axes[0].set_title('neg_root_mean_squared_error', fontsize=7, fontweight="bold")
axes[0].tick_params(labelsize=6)
axes[0].set_xlabel("")

sns.boxplot(
    x=cv_scores['test_neg_root_mean_squared_error'],
    ax=axes[1]
)
axes[1].set_title('neg_root_mean_squared_error', fontsize=7, fontweight="bold")
axes[1].tick_params(labelsize=6)
axes[1].set_xlabel("")

fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Distribución error de validación cruzada', fontsize=10,
             fontweight="bold")

# Diagnóstico errores (residuos) de las predicciones de validación cruzada
# ==============================================================================
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
import statsmodels.api as sm

# Validación cruzada
# ==============================================================================
cv = KFold(n_splits=5, random_state=123, shuffle=True)
cv_prediccones = cross_val_predict(
    estimator=pipe,
    X=X_train,
    y=y_train,
    cv=cv
)

# Gráficos
# ==============================================================================
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 5))

axes[0, 0].scatter(y_train, cv_prediccones, edgecolors=(0, 0, 0), alpha=0.4)
axes[0, 0].plot(
    [y_train.min(), y_train.max()],
    [y_train.min(), y_train.max()],
    'k--', color='black', lw=2
)
axes[0, 0].set_title('Valor predicho vs valor real', fontsize=10, fontweight="bold")
axes[0, 0].set_xlabel('Real')
axes[0, 0].set_ylabel('Predicción')
axes[0, 0].tick_params(labelsize=7)

axes[0, 1].scatter(list(range(len(y_train))), y_train - cv_prediccones,
                   edgecolors=(0, 0, 0), alpha=0.4)
axes[0, 1].axhline(y=0, linestyle='--', color='black', lw=2)
axes[0, 1].set_title('Residuos del modelo', fontsize=10, fontweight="bold")
axes[0, 1].set_xlabel('id')
axes[0, 1].set_ylabel('Residuo')
axes[0, 1].tick_params(labelsize=7)

sns.histplot(
    data=y_train - cv_prediccones,
    stat="density",
    kde=True,
    line_kws={'linewidth': 1},
    color="firebrick",
    alpha=0.3,
    ax=axes[1, 0]
)

axes[1, 0].set_title('Distribución residuos del modelo', fontsize=10,
                     fontweight="bold")
axes[1, 0].set_xlabel("Residuo")
axes[1, 0].tick_params(labelsize=7)

sm.qqplot(
    y_train - cv_prediccones,
    fit=True,
    line='q',
    ax=axes[1, 1],
    color='firebrick',
    alpha=0.4,
    lw=2
)
axes[1, 1].set_title('Q-Q residuos del modelo', fontsize=10, fontweight="bold")
axes[1, 1].tick_params(labelsize=7)

fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Diagnóstico residuos', fontsize=12, fontweight="bold")

plt.show()

# Validación cruzada repetida paralelizada (multicore)
# ==============================================================================
from sklearn.model_selection import RepeatedKFold

cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=123)
cv_scores = cross_val_score(
    estimator=pipe,
    X=X_train,
    y=y_train,
    scoring='neg_root_mean_squared_error',
    cv=cv,
    n_jobs=-1  # todos los cores disponibles
)

print(f"Média métricas de validación cruzada: {cv_scores.mean()}")

predicciones = pipe.predict(X_test)
# Se crea un dataframe con las predicciones y el valor real
df_predicciones = pd.DataFrame({'precio': y_test, 'prediccion': predicciones})
print(df_predicciones.head())
