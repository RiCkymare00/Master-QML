import numpy as np
from pysr import PySRRegressor

# Genera i primi 100 termini della sequenza di Fibonacci
def fibonacci(n):
    fib = [0, 1]
    for _ in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib

# Costanti matematiche da includere nella regressione
phi = (1 + np.sqrt(5)) / 2
psi = (1 - np.sqrt(5)) / 2
sqrt5 = np.sqrt(5)

# Input: n = 0, 1, ..., 99
X = np.arange(100).reshape(-1, 1)
y = np.array(fibonacci(100))

# Avvia il regressore simbolico
model = PySRRegressor(
    niterations=1000,
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["exp", "log", "sqrt", "sin", "cos"],
    extra_sympy_mappings={"phi": phi, "psi": psi, "sqrt5": sqrt5},
    constraints={"^": (1, 8)},  # limita le potenze a esponenti ragionevoli
    model_selection="best",
    loss="loss(x, y) = (x - y)^2",
    maxsize=30,
    verbosity=1,
    multithreading=False,  # spesso più stabile disabilitare il multithreading
    random_state=42,
    deterministic=True,
)

model.fit(X, y)

# Stampa la formula trovata
print(model)
