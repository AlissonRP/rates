#%%
from model.Kuma import Kuma
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from model.utils import bar_ploto
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


df = pd.read_csv("data/desmatamento.prop.csv").drop(["Unnamed: 0"], axis=1)
df.rename(columns={"desmatamento_pro": "prop"}, inplace=True)
df['prop'] = np.where(df['prop'] >= 1, 0.99, df['prop'])
df['prop'] = np.where(df['prop'] <= 0, 0.01, df['prop'])
prop = df['prop']
kuma = Kuma(0.5, 0.5)
theta0 = [0.5, 1]

kuma.fit(theta = [0.5, 1], data = prop,  change=True)

kuma.plot(prop)

kuma.metrics(prop)
#%%
def AD_beta(x, a, b):
        cdf =  stats.beta.cdf(np.sort(x), a = a, b = b)
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
        ad = a2*(1+0.75/len(x) + 2.25/(len(x)**2))
        return ad

def AD_norm(x, loc, scale):
        cdf = stats.norm.cdf(np.sort(x), loc = loc, scale = scale)
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
        ad = a2*(1+0.75/len(x) + 2.25/(len(x)**2))
        return ad

def AIC(likelihood, num_parameters):
    aic = 2 * likelihood + 2 * num_parameters
    return aic

def BIC(likelihood, num_parameters, length):
    bic = 2 * likelihood + num_parameters * np.log(length)
    return bic

def CAIC(likelihood, num_parameters, length):
    aicc = 2 * likelihood + 2 * num_parameters * length / (
        length - num_parameters - 1
    )
    return aicc

#%%
def llnorm(par, data):
    mu, sigma = par
    ll = -np.sum(np.log(2*np.pi*(sigma**2))/2 + ((data-mu)**2)/(2 * (sigma**2)))
    return ll

def betaNLL(par, data):
    """
    Negative log likelihood function for beta
    <param>: list for parameters to be fitted.
    <args>: 1-element array containing the sample data.

    Return <nll>: negative log-likelihood to be minimized.
    """

    a, b = par
    pdf = stats.beta.pdf(data,a,b,loc=0,scale=1)
    lg = np.log(pdf)
    mask = np.isfinite(lg)
    nll = -lg[mask].sum()
    return nll
#%%

mle = minimize(llnorm, x0=theta0, method="Nelder-Mead", args=(prop))
stats.cramervonmises(prop, 'beta',  args=(mle.x[0], mle.x[1]))
stats.kstest(prop, 'beta',  args=(mle.x[0], mle.x[1]))
AD_norm(prop, mle.x[0], mle.x[1])
AIC(mle.fun, len(mle.x))
BIC(mle.fun, len(mle.x), len(prop))
CAIC(mle.fun, len(mle.x), len(prop))
#%%
mle = minimize(betaNLL, x0=theta0, method="Nelder-Mead", args=(prop))
stats.cramervonmises(prop, 'beta',  args=(mle.x[0], mle.x[1]))
stats.kstest(prop, 'norm',  args=(mle.x[0], mle.x[1]))
AD_beta(prop, mle.x[0], mle.x[1])
AIC(mle.fun, len(mle.x))
BIC(mle.fun, len(mle.x), len(prop))
CAIC(mle.fun, len(mle.x), len(prop))
# stats.cramervonmises(prop, 'beta')
#%%