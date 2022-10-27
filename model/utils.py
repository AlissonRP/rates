
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

def bar_ploto(df, variable="Estado", metric='mean'):
    """
    metric: pode ser mean, min, max etc, e interessante pois tem municipios com prop=1
    """
    desmatamento = (df.groupby(variable, as_index=False)
    .agg(metric)
    .sort_values(['prop'], ascending=False))


    desmatamento['prop'] = round(desmatamento['prop'], 2)

    sns.set_style("whitegrid")
    ax = sns.barplot(data = desmatamento, x=variable, y="prop")
    # plt.xlim([-0.009, 1.009])
    for i in ax.containers:
        ax.bar_label(i, )




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