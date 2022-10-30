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

#%%

mle = minimize(llnorm, x0=theta0, method="Nelder-Mead", args=(prop))
res = stats.cramervonmises(prop, 'norm',  args=(mle.x[0], mle.x[1]))
res1 = stats.kstest(prop, 'norm',  args=(mle.x[0], mle.x[1]))
AD_norm(prop, mle.x[0], mle.x[1])
AIC(mle.fun, len(mle.x))
BIC(mle.fun, len(mle.x), len(prop))
CAIC(mle.fun, len(mle.x), len(prop))
criterios = kuma.metrics(prop)
criterios_kuma = [criterios['AIC'], criterios['AICc'], criterios['BIC'], criterios['cramer_vonmises'], criterios['AD'], criterios['KS']]
criterios_normal = ([AIC(mle.fun, len(mle.x)), BIC(mle.fun, len(mle.x), len(prop)), CAIC(mle.fun, len(mle.x), len(prop)),
                    AD_norm(prop, mle.x[0], mle.x[1]), res.statistic, res1.statistic])
#%%
mle = minimize(betaNLL, x0=theta0, method="Nelder-Mead", args=(prop))
res = stats.cramervonmises(prop, 'beta',  args=(mle.x[0], mle.x[1]))
res1 = stats.kstest(prop, 'beta',  args=(mle.x[0], mle.x[1]))
AD_beta(prop, mle.x[0], mle.x[1])
AIC(mle.fun, len(mle.x))
BIC(mle.fun, len(mle.x), len(prop))
CAIC(mle.fun, len(mle.x), len(prop))
criterios_beta = ([AIC(mle.fun, len(mle.x)), BIC(mle.fun, len(mle.x), len(prop)), CAIC(mle.fun, len(mle.x), len(prop)),
                    AD_beta(prop, mle.x[0], mle.x[1]), res.statistic, res1.statistic])
# stats.cramervonmises(prop, 'beta')
#%%
bar_ploto(df)
# %%
