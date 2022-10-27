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
        if not ((0 < x) & (x < 1)).all():
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
        def verossimilhanca(theta, x):
            alpha, beta = theta
            return -(
                (len(x) * np.log(alpha))
                + (len(x) * np.log(beta))
                + ((alpha - 1) * np.sum(np.log(x)))
                + ((beta - 1) * np.sum(np.log(1 - x**alpha)))
            )
        theta0 = theta
        mle = minimize(verossimilhanca, x0=theta0, method="Nelder-Mead", args=(data)).x
        
        if change == True:
            self.alpha = mle[0]
            self.beta = mle[1]
            self.vero(data)
        return self

    def vero(self, x):
        self.likelihood = -(
                            (len(x) * np.log(self.alpha))
                            + (len(x) * np.log(self.beta))
                            + ((self.alpha - 1) * np.sum(np.log(x)))
                            + ((self.beta - 1) * np.sum(np.log(1 - x**self.alpha)))
                        )
        return self

    def AIC(self):
        self.aic = 2 * self.likelihood + 2 * self.num_parameters
        return self.aic

    def BIC(self, x):
        self.bic = 2 * self.likelihood + self.num_parameters * np.log(len(x))
        return self.bic

    def CAIC(self, x):
        self.aicc = 2 * self.likelihood + 2 * self.num_parameters * len(x) / (
            len(x) - self.num_parameters - 1
        )
        return self.aicc

    def CVM(self,x):
        cdf = self.cumulative(np.sort(x))
        y = stats.norm.ppf(cdf)
        u = stats.norm.cdf((y-np.mean(y))/np.std(y))
        aux = np.concatenate(np.fromiter(((element - (2*np.where(np.isclose(u, element))[0] -1)/(2*len(x)))**2 for element in u), object))
        w2 = sum(aux) + 1/(12*len(x))
        self.cramermises = w2*(1+0.5/len(x))
        return self.cramermises
    
    def AD(self,x):
        cdf = self.cumulative(np.sort(x))
        y = stats.norm.ppf(cdf)
        u = stats.norm.cdf((y-np.mean(y))/np.std(y))
        # aux = (np.concatenate(np.fromiter(((2*np.where(np.isclose(u, element))[0] -1)*np.log(element) + 
        #                                     (2*len(x) - 2*np.where(np.isclose(u, element))[0] +1)*np.log(1-element) 
        #                                     for element in u), object)))
        aux = np.zeros(len(y))
        i = 0
        for u_ in u:
            aux[i] = (2*i-1)*np.log(u_) + (2*len(x) - 2*i+1)*np.log(1-u_)
            i = i + 1
        a2 = -len(x) -(1/len(x))*sum(aux) 
        self.ad = a2*(1+0.75/len(x) + 2.25/(len(x)**2))
        return self.ad

    def KS(self,x):
        cdf = self.cumulative(np.sort(x))
        aux1 = np.zeros(len(cdf))
        aux2 = np.zeros(len(cdf))
        i = 0
        for fda in cdf:
            aux1[i] = fda -(i-1)/len(x)
            aux2[i] = i/len(x) - fda
            i = i + 1
        self.ks = max([max(aux1), max(aux2), 0])
        return self.ks
    
    def metrics(self, x = [], theta = [], change = False):
        try:
            return {    
                        'alpha' : self.alpha,
                        'beta' : self.beta,
                        'likelihood' : self.likelihood,
                        'AIC' : self.aic,
                        'AICc' : self.aicc,
                        'BIC' : self.bic,
                        'cramer_vonmises' : self.cramermises,
                        'AD' : self.ad,
                        'KS' : self.ks
                    }
        except:
            self.fit(theta, x, change)
            self.vero(x).AIC()
            self.CAIC(x)
            self.BIC(x)
            self.CVM(x)
            self.AD(x)
            self.KS(x)
            return {    
                        'alpha' : self.alpha,
                        'beta' : self.beta,
                        'likelihood' : self.likelihood,
                        'AIC' : self.aic,
                        'AICc' : self.aicc,
                        'BIC' : self.bic,
                        'cramer_vonmises' : self.cramermises,
                        'AD' : self.ad,
                        'KS' : self.ks
                    }
    
    def plot(self, data, legend_local="right"):
        """legend_local: local da legenda, pois a distribuicao pode ter ambas tipo de assimetria,
        entao pode ser necessario mover a legenda para esquerda
        """
        ax = sns.histplot(data, kde=True, stat="density", bins=20)
        ax.lines[0].set_color("orange")
        x = np.random.uniform(0, 1, len(data))
        ax = sns.kdeplot(self.quantile(x))
        plt.xlim([-0.009, 1.009])
        plt.ylabel("Densidade")
        plt.legend(
            title="Tipo", loc="upper " + legend_local, labels=["Dados", "Distribuição"]
        )

#### TESTES  MANUAIS#####
# teste = Kuma(alpha=5, beta=3)


#%%
# x = np.random.uniform(0, 1, 5000)


# sns.kdeplot(teste.quantile(x))


# #%%


# x = np.random.uniform(0, 1, 5000)
# y = teste.quantile(x)


# #%

# #%%%


#%%
# teste.fit([1, 2], y, change=True)

# sns.histplot(y)

#%%
# teste.plot(y)

# # %%

