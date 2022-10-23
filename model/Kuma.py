#%%
import numpy as np
from math import gamma
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize


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
            * (1 - x**self.alpha) ** (self.beta - 1)
        )

    def quantile(self, u):
        if (np.all(0 < u) and np.all(u < 1)) == False:
            raise ValueError("u must be between 0 and 1")
        return (1 - (1 - u) ** (1 / self.beta)) ** (1 / self.alpha)

    def cumulative(self, x):
        if not 0 < x < 1:
            raise ValueError("X must be between 0 and 1")
        return 1 - (1 - (x**self.alpha)) ** self.beta

    def mean(self):
        return (self.beta * gamma(1 + (1 / self.alpha)) * gamma(self.beta)) / gamma(
            1 + (1 / self.alpha) + self.beta
        )

    def log_vero(self, x):
        return np.sum(np.log(self.pdf(x)))

    def fit(self, theta, data, change=False):
        """
        theta: chute inicial
        change: Se for True, entao os parametros da classe vão ser substituidos pelos estimados
        """

        def vero(theta, x):
            alpha, beta = theta
            return -(
                (len(x) * np.log(alpha))
                + (len(x) * np.log(beta))
                + ((alpha - 1) * np.sum(np.log(x)))
                + ((beta - 1) * np.sum(np.log(1 - x**alpha)))
            )

        theta0 = theta
        mle = minimize(vero, x0=theta0, method="Nelder-Mead", args=(data)).x
        if change == True:
            self.alpha = mle[0]
            self.beta = mle[1]
        return {"alpha_est": mle[0], "beta_est": mle[1]}

    def plot(self, data):
        sns.distplot(data, bins=20, hist_kws={"edgecolor": 'black'})
        x = np.random.uniform(0, 1, len(data))
        sns.kdeplot(teste.quantile(x))
        plt.ylabel("Densidade")
        plt.legend(title='Tipo', loc='upper right', labels=['Dados', 'Distribuição'])


#### TESTES  MANUAIS#####
#%%
teste = Kuma(alpha=5, beta=3)


#%%
x = np.random.uniform(0, 1, 5000)


sns.kdeplot(teste.quantile(x))


#%%


x = np.random.uniform(0, 1, 5000)
y = teste.quantile(x)


#%

#%%%


#%%
teste.fit([1, 2], y, change=True)

#%%






#sns.histplot(y)

#%%
teste.plot(y)
