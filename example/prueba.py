import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(1234)
x = np.random.randn(30)
y = -0.5 * x ** 4 + 2 * x ** 3 - 3 + np.random.randn(len(x))

print(x.reshape(-1, 1))
print(x, y)

# Ajustar el modelo a este datasets utilizan LinearRegression
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)
print(f"Coeficiente de determinación: {model.score(x.reshape(-1, 1), y):0.4f}")

x_test = np.linspace(x.min(), x.max(), 200)
y_test = model.predict(x_test.reshape(-1, 1))

print(x_test, y_test)


def plot_prediction(x, y, x_test, y_test, model):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True, tight_layout=True)
    ax[0].scatter(x, y, s=50, label='Datos', alpha=0.5)
    ax[0].plot(x_test, y_test, label='Modelo', lw=1, color='purple')
    ax[0].legend()
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[1].scatter(y, y - model.predict(x.reshape(-1, 1)), alpha=0.5)
    ax[1].axhline(0, c='k', ls='--', alpha=0.5)
    ax[1].set_xlabel('y')
    ax[1].set_ylabel('residuos')


plot_prediction(x, y, x_test, y_test, model)
plt.show()

print(f"MSE: {mean_squared_error(y, model.predict(x.reshape(-1, 1))):0.4f}")

# Regresion Polinomial de grado M
features = PolynomialFeatures(degree=2)

# Regresor lineal con pipelines

model = make_pipeline(PolynomialFeatures(degree=3),
                      LinearRegression(fit_intercept=False))

print(model)

model.fit(x.reshape(-1, 1), y)
print(f"MSE : {mean_squared_error(y, model.predict(x.reshape(-1, 1))):0.4f}")
y_test = model.predict(x_test.reshape(-1, 1))

plot_prediction(x, y, x_test, y_test, model)

plt.show()
