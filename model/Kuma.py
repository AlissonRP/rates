#%%
import numpy as np
from math import gamma
import matplotlib.pyplot as plt
import seaborn as sns


class Kuma:
    def __init__(self, alpha, beta):
        if len([i for i in [alpha, beta] if i < 0]) > 0:
            raise ValueError("alpha and beta must be greater than zero")
        self.alpha = alpha
        self.beta = beta

    def pdf(self, x):
        if (np.all(0 < x) and np.all(x < 1)) == False:
            raise ValueError("X must be between 0 and 1")
        return (
            self.alpha
            * self.beta
            * x ** (self.alpha - 1)
            * (1 - x ** self.alpha) ** (self.beta - 1)
        )

    def quantile(self, u):
        if (np.all(0 < u) and np.all(u < 1)) == False:
            raise ValueError("u must be between 0 and 1")
        return (1 - (1 - u) ** (1 / self.beta)) ** (1 / self.alpha)

    def cumulative(self, x):
        if not 0 < x < 1:
            raise ValueError("X must be between 0 and 1")
        return 1 - (1 - (x ** self.alpha)) ** self.beta

    def mean(self):
        return (self.beta * gamma(1 + (1 / self.alpha)) * gamma(self.beta)) / gamma(
            1 + (1 / self.alpha) + self.beta
        )

    def log_vero(self, x):
        return np.sum(np.log(self.pdf(x)))


#%%
teste = Kuma(alpha=5, beta=1)


#%%
x = np.random.uniform(0, 1, 5000)


sns.kdeplot(teste.quantile(x))
