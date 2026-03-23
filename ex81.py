# Experiment 8.1: Gradient-Based Optimization Analysis

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def J(w):
    w1, w2 = w
    return (w1**2 + w2 - 11)**2 + (w1 + w2**2 - 7)**2

def grad_J(w):
    w1, w2 = w
    dw1 = 4*w1*(w1**2 + w2 - 11) + 2*(w1 + w2**2 - 7)
    dw2 = 2*(w1**2 + w2 - 11) + 4*w2*(w1 + w2**2 - 7)
    return np.array([dw1, dw2])

def GD(init, lr=0.01, steps=100):
    w = init.copy()
    path = [w.copy()]
    loss = []
    for _ in range(steps):
        w -= lr * grad_J(w)
        path.append(w.copy())
        loss.append(J(w))
    return np.array(path), loss

def Nesterov(init, lr=0.01, gamma=0.9, steps=100):
    w = init.copy()
    v = np.zeros_like(w)
    path = [w.copy()]
    loss = []
    for _ in range(steps):
        lookahead = w - gamma * v
        g = grad_J(lookahead)
        v = gamma * v + lr * g
        w -= v
        path.append(w.copy())
        loss.append(J(w))
    return np.array(path), loss

def Adagrad(init, lr=0.4, steps=100):
    w = init.copy()
    G = np.zeros_like(w)
    eps = 1e-8
    path = [w.copy()]
    loss = []
    for _ in range(steps):
        g = grad_J(w)
        G += g**2
        w -= lr * g / (np.sqrt(G) + eps)
        path.append(w.copy())
        loss.append(J(w))
    return np.array(path), loss

def RMSProp(init, lr=0.01, beta=0.9, steps=100):
    w = init.copy()
    Eg = np.zeros_like(w)
    eps = 1e-8
    path = [w.copy()]
    loss = []
    for _ in range(steps):
        g = grad_J(w)
        Eg = beta * Eg + (1 - beta) * (g**2)
        w -= lr * g / (np.sqrt(Eg) + eps)
        path.append(w.copy())
        loss.append(J(w))
    return np.array(path), loss

def Adam(init, lr=0.05, steps=100):
    w = init.copy()
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    path = [w.copy()]
    loss = []
    for t in range(1, steps + 1):
        g = grad_J(w)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g**2)
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        w -= lr * mhat / (np.sqrt(vhat) + eps)
        path.append(w.copy())
        loss.append(J(w))
    return np.array(path), loss

init = np.array([-4.0, 4.0])
optimizers = {
    "GD": GD,
    "Nesterov": Nesterov,
    "Adagrad": Adagrad,
    "RMSProp": RMSProp,
    "Adam": Adam
}

paths = {}
losses = {}
for name, opt in optimizers.items():
    paths[name], losses[name] = opt(init)

# Plot Contours
x = np.linspace(-6, 6, 200)
y = np.linspace(-6, 6, 200)
X, Y = np.meshgrid(x, y)
Z = J([X, Y])

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, 40)
for name, p in paths.items():
    plt.plot(p[:, 0], p[:, 1], label=name)
plt.legend()
plt.title("Optimizer Paths on Loss Contour")
plt.show()

# Plot Loss Curves
plt.figure()
for name, l in losses.items():
    plt.plot(l, label=name)
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss vs Iterations")
plt.show()
