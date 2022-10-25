#%%
import numpy as np
from math import gamma
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from inspect import signature
import scipy.stats as stats


class Kuma:
    def __init__(self, alpha, beta):
        """
        alpha e beta sao os parametros da distribuicao
        """
        if len([i for i in [alpha, beta] if i < 0]) > 0:
            raise ValueError("alpha and beta must be greater than zero")
        self.alpha = alpha
        self.beta = beta
        self.num_parameters = len(signature(Kuma).parameters)

    def pdf(self, x):
        """
        x: valor a se calcular a densidade
        """
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
        return self

    def vero(self, x):
        self.log_vero = -(
            (len(x) * np.log(self.alpha))
            + (len(x) * np.log(self.beta))
            + ((self.alpha - 1) * np.sum(np.log(x)))
            + ((self.beta - 1) * np.sum(np.log(1 - x**self.alpha)))
        )
        return self

    def AIc(self):
        self.AIC = 2 * self.log_vero + 2 * self.num_parameters
        return self.AIC

    def BIc(self, x):
        self.BIC = 2 * self.log_vero + self.num_parameters * log(len(x))
        return self.BIC

    def AIcc(self, x):
        self.AICC = 2 * self.log_vero + 2 * self.num_parameters * len(x) / (
            len(x) - self.num_parameters - 1
        )
        return self.AICC

    def cramervonmises(self, x):
        cdf = self.cumulative(np.sort(x))
        y = stats.norm.ppf(cdf)
        u = stats.norm.cdf((y - np.mean(y)) / np.std(y))
        aux = [
            (element - (2 * np.where(np.isclose(u, element))[0]) / (2 * len(x))) ** 2
            for element in u
        ]
        w2 = sum(aux) + 1 / (12 * len(x))
        self.cramermises = w2 * (1 + 0.5 / len(x))
        return aux, w2, u

    def plot(self, data, legend_local="right"):
        """
        legend_local: local da legenda, pois a distribuicao pode ter ambas tipo de assimetria,
        entao pode ser necessario mover a legenda para esquerda
        """
        ax = sns.histplot(data, kde=True, stat="density", bins=20)
        ax.lines[0].set_color("orange")
        x = np.random.uniform(0, 1, len(data))
        ax = sns.kdeplot(self.quantile(x))
        plt.ylabel("Densidade")
        plt.legend(
            title="Tipo", loc="upper " + legend_local, labels=["Dados", "Distribuição"]
        )


#### TESTES  MANUAIS#####
# teste = Kuma(alpha=5, beta=3)


#%%
# x = np.random.uniform(0, 1, 5000)


# sns.kdeplot(teste.quantile(x))


#%%


# x = np.random.uniform(0, 1, 5000)
# y = teste.quantile(x)


#%

#%%%


#%%
# teste.fit([1, 2], y, change=True)

#%%


# sns.histplot(y)

#%%
# teste.plot(y)

# %%
