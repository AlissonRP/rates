# Pacote necessario para fazer a maximiza??o

rm(list=ls())

library(maxLik)

#####################################################################
# Fun??o de densidade da EGSGu
pdf_EGSGu <- function(par,x){
  
  a = par[1]
  b = par[2]
  f = a*b*exp(-x-exp(-x))*( 1-exp(-exp(-x)))^(a-1)*((1-( 1-exp(-exp(-x)))^a))^(b-1)
  f
}

# Fun??o de distribui??o da EGSGu
cdf_EGSGu <- function(par,x){
  a = par[1]
  b = par[2]
  Fx = ((1-( 1-exp(-exp(-x)))^a))^b
  Fx
}

#####################################################################
pdf_egep <- function(par,x){
  xi <- par[1]
  sigma <- 1
  a <- par[2]
  b <- par[3]
  p1 <- 1+xi*x/sigma
  g <- a*b*p1^(-(a+xi)/xi)*(1-p1^(-a/xi))^(b-1)/sigma
  g
}

# Fun??o de densidade da EGEP
cdf_egep <- function(par,x){
  xi <- par[1]
  sigma <- 1
  a <- par[2]
  b <- par[3]
  p1 <- 1+xi*x/sigma
  Fx <- (1-p1^(-a/xi))^b
  Fx
}



#####################################################################
# Fun??o de densidade da GGP
pdf_ggp <- function(par,x){
  xi = par[1]
  sigma = par[2]
  a = par[3]
  z=1+xi*x/sigma
  f = ((log(z))^(a-1)*z^(-1/xi-1))/(gamma(a)*sigma*xi^(a-1))
  f
}

# Fun??o de distribui??o da GGP
cdf_ggp <- function(par,x){
  xi = par[1]
  sigma = par[2]
  a = par[3]
  z=1+xi*x/sigma
  Fx = pgamma(1/xi*log(z), a) 
  Fx
}


#####################################################################
# Fun??o de densidade da KGP
pdf_kumagp <- function(par,x){
  xi = par[1]
  sigma = par[2]
  a = par[3]
  b = par[4]
  z = 1 + xi*x/sigma
  G = 1 - z^(-1/xi)
  g = z^(-1/xi-1)/sigma
  f = a*b*g*G^(a-1)*( 1 - G^a )^(b-1)
  f
}

# Fun??o de distribui??o da KGP
cdf_kgp <- function(par,x){
  xi = par[1]
  sigma = par[2]
  a = par[3]
  b = par[4]
  z = 1 + xi*x/sigma
  G = 1 - z^(-1/xi)
  g = z^(-1/xi-1)/sigma
  Fx = 1 - (1 - G^a)^b
  Fx
}

#####################################################################
# Fun??o de densidade da BGP
pdf_betagp <- function(par,x){
  xi = par[1]
  sigma = par[2]
  alpha = par[3]
  beta1 = par[4]
  z = 1 + xi*x/sigma
  G = 1 - z^(-1/xi)
  g = z^(-1/xi-1)/sigma
  f = g*G^(alpha-1)*z^(-1/xi*(beta1-1))/beta(alpha,beta1)
  f
}

# Fun??o de distribui??o da BGP
cdf_bgp <- function(par,x){
  xi = par[1]
  sigma = par[2]
  alpha = par[3]
  beta1 = par[4]
  z = 1 + xi*x/sigma
  G = 1 - z^(-1/xi)
  Fx = pbeta(G, alpha, beta1)
  Fx
}

####################################################
# KumaBXII - Probability density function.
pdf_kumaBXII <- function(par,x){
  a = par[1]
  b = par[2]
  c = par[3]  
  k = par[4]
  s = par[5]
  G= 1-(1+(x/s)^c)^(-k)
  g= c*k*(s^(-c))*(x^(c-1))*(1+(x/s)^c)^(-k-1)
  f = a*b*g*(G^(a-1))*(1-G^a)^(b-1)  
  f
}

# KumaGP - Cumulative distribution function.
cdf_kumaBXII <- function(par,x){
  
  a = par[1]
  b = par[2]
  c = par[3]  
  k = par[4]
  s = par[5]
  G = 1-(1+(x/s)^c)^(-k)
  Fx = 1-(1-G^a)^b
  Fx
}

###############################################
# BMW - Probability density function.
pdf_BMW <- function(par,x){
  alpha = par[1]
  gama = par[2]
  lambda = par[3]
  a = par[4]
  b = par[5]
  z = alpha*x^gama*exp(lambda*x)
  G = 1 - exp(-z)
  g = alpha*x^(gama-1)*(gama+lambda*x)*exp(lambda*x-z)
  f = g*G^(a-1)*(1-G)^(b-1)/beta(a,b)
  f
}

# BMW - Cumulative distribution function.
cdf_BMW <- function(par,x){
  alpha = par[1]
  gama = par[2]
  lambda = par[3]
  a = par[4]
  b = par[5]
  z = alpha*x^gama*exp(lambda*x)
  G = 1 - exp(-z)
  f = pbeta(G, a, b)
  f
}

#####################################################################


# Fun??o de densidade da Exp-Weilbul
pdf_EW <- function(par,x){
  
  a = par[1]
  alpha = par[2]
  gama = par[3]
  
  f = a*alpha*gama*x^(gama-1)*exp(-alpha*x^gama)*(1- exp(-alpha*x^gama))^(a-1)
  f
}

# Fun??o de distribui??o da Exp-Weilbull
cdf_EW <- function(par,x){
  a = par[1]
  alpha = par[2]
  gama = par[3]
  
  Fx = (1- exp(-alpha*x^gama))^a
  Fx
}
#####################################################################


# Fun??o de densidade da Exp-Exponential(EE)
pdf_EE <- function(par,x){
  
  
  alpha = par[1]
  lambda = par[2]
  f = alpha*lambda*exp(-lambda*x)*(1-exp(-lambda*x))^(alpha-1) 
  f
  
}

# Fun??o de distribui??o da Exp-Exponential(EE)
cdf_EE <- function(par,x){
  alpha = par[1]
  lambda = par[2]
  
  Fx = (1-exp(-lambda*x))^alpha
  Fx
}


pdf_BMW <- function(par,x){
  alpha = par[1]
  gama = par[2]
  lambda = par[3]
  a = par[4]
  b = par[5]
  z = alpha*x^gama*exp(lambda*x)
  G = 1 - exp(-z)
  g = alpha*x^(gama-1)*(gama+lambda*x)*exp(lambda*x-z)
  f = g*G^(a-1)*(1-G)^(b-1)/beta(a,b)
  f
}

#####################################################################
# Fun??o de densidade da KW
pdf_kw <- function(par,x){
  alpha = par[1]
  beta1 = par[2]
  a = par[3]
  b = par[4]
  g = dweibull(x,shape=alpha,scale=beta1)
  G = pweibull(x,shape=alpha,scale=beta1)  
  f = a*b*g*G^(a-1)*( 1 - G^a )^(b-1)
  f
}


# Fun??o de distribui??o da KW
cdf_kw <- function(par,x){
  alpha = par[1]
  beta1 = par[2]
  a = par[3]
  b = par[4]
  G = pweibull(x,shape=alpha,scale=beta1)  
  Fx = 1 - (1 - G^a)^b
  Fx
}

#####################################################################
# Fun??o de densidade da BetaW
pdf_betaw <- function(par,x){
  a = par[1]
  b = par[2]
  alpha = par[3]
  beta1 = par[4]
  g = dweibull(x,shape=a,scale=b)
  G = pweibull(x,shape=a,scale=b)  
  f = G^(alpha-1)*(1-G)^(beta1-1)*g/beta(alpha,beta1)
  f
}

# Fun??o de distribui??o da BetaW
cdf_betaw <- function(par,x){
  a = par[1]
  b = par[2]
  alpha = par[3]
  beta1 = par[4]
  G = pweibull(x,shape=a,scale=b)  
  Fx = pbeta(G, alpha, beta1)
  Fx
}


#####################################################################
# Fun??o de densidade da Gamma-W
pdf_gammaw <- function(par,x){
  a = par[1]
  alpha = par[2]
  beta1 = par[3]
  g = dweibull(x,shape=alpha,scale=beta1)
  G = pweibull(x,shape=alpha,scale=beta1)  
  f = (-log(1-G))^(a-1)*g/gamma(a)
  f
}

# Fun??o de distribui??o da Gamma-W
cdf_gammaw <- function(par,x){
  a = par[1]
  alpha = par[2]
  beta1 = par[3]
  G = pweibull(x,shape=alpha,scale=beta1)  
  Fx = pgamma(-log(1-G),shape=a)
  Fx
}


#####################################################################
# Fun??o de densidade da EG-W
pdf_EGW <- function(par,x){
  a = par[1]
  b = par[2]
  alpha = par[3]
  beta1 = par[4]
  g = dweibull(x,shape=alpha,scale=beta1)
  G = pweibull(x,shape=alpha,scale=beta1)  
  f = a*b*(1-G)^(a-1)*(1-(1-G)^a)^(b-1)
  f
}

# Fun??o de distribui??o da EG-W
cdf_EGW <- function(par,x){
  a = par[1]
  b = par[2]
  alpha = par[3]
  beta1 = par[4]
  G = pweibull(x,shape=alpha,scale=beta1)  
  Fx = (1-(1-G)^a)^b
  Fx
}


pdf_kgama <- function(par,x){
  alpha = par[1]
  beta1 = par[2]
  a = par[3]
  b = par[4]
  g = dgamma(x,shape=alpha,scale=beta1)
  G = pgamma(x,shape=alpha,scale=beta1)  
  f = a*b*g*G^(a-1)*( 1 - G^a )^(b-1)
  f
}


# Fun??o de distribui??o da K-gamma
cdf_kgama <- function(par,x){
  alpha = par[1]
  beta1 = par[2]
  a = par[3]
  b = par[4]
  G = pgamma(x,shape=alpha,scale=beta1)  
  Fx = 1 - (1 - G^a)^b
  Fx
}

#####################################################################
# Fun??o de densidade da Beta-Gamma
pdf_betaG <- function(par,x){
  a = par[1]
  b = par[2]
  alpha = par[3]
  beta1 = par[4]
  g = dgamma(x,shape=a,scale=b)
  G = pgamma(x,shape=a,scale=b)  
  f = G^(alpha-1)*(1-G)^(beta1-1)*g/beta(alpha,beta1)
  f
}

# Fun??o de distribui??o da Beta-Gamma
cdf_betaG <- function(par,x){
  a = par[1]
  b = par[2]
  alpha = par[3]
  beta1 = par[4]
  G = pgamma(x,shape=a,scale=b)  
  Fx = pbeta(G, alpha, beta1)
  Fx
}

#####################################################################
# Fun??o de densidade da Gamma-gama
pdf_gammaG <- function(par,x){
  a = par[1]
  alpha = par[2]
  beta1 = par[3]
  g = dgamma(x,shape=alpha,scale=beta1)
  G = pgamma(x,shape=alpha,scale=beta1)  
  f = (-log(1-G))^(a-1)*g/gamma(a)
  f
}

# Fun??o de distribui??o da Gamma-gama
cdf_gammaG <- function(par,x){
  a = par[1]
  alpha = par[2]
  beta1 = par[3]
  G = pgamma(x,shape=alpha,scale=beta1)  
  Fx = pgamma(-log(1-G),shape=a)
  Fx
}

#####################################################################
# Fun??o de densidade da EG-Gamma
pdf_EGgama <- function(par,x){
  a = par[1]
  b = par[2]
  alpha = par[3]
  beta1 = par[4]
  g = dgamma(x,shape=alpha,scale=beta1)
  G = pgamma(x,shape=alpha,scale=beta1)  
  f = a*b*(1-G)^(a-1)*(1-(1-G)^a)^(b-1)
  f
}

# Fun??o de distribui??o da EG-Gama
cdf_EGgama <- function(par,x){
  a = par[1]
  b = par[2]
  alpha = par[3]
  beta1 = par[4]
  G = pgamma(x,shape=alpha,scale=beta1)  
  Fx = (1-(1-G)^a)^b
  Fx
}



####################################################################
# Fun??o de densidade da Erf_We
pdf_Erf_We <- function(par,x){
  alpha = par[1]
  gama = par[2]
  g = alpha*gama*x^(gama-1)*exp(-alpha*x^gama)
  G = 1- exp(-alpha*x^gama)
  f = 2*g*exp(-(G/(1-G))^2)/(sqrt(pi)*(1-G)^2)
  f
}

# Fun??o de distribui??o da Erf_We
cdf_Erf_We <- function(par,x){
  alpha = par[1]
  gama = par[2]
  g = alpha*gama*x^(gama-1)*exp(-alpha*x^gama)
  G = 1- exp(-alpha*x^gama)
  Fx =  2*pnorm((G/(1-G))*sqrt(2))-1
  Fx
}

# Log-verossimilhan?a da EGSGu_Pr_We
mlog_Erf_We <- function(par,x){
  s <- sum(log(pdf_Erf_We(par=par,x=x)))
  s
}



####################################################################
# Fun??o de densidade da Erf_Exp
pdf_Erf_Exp <- function(par,x){
  alpha = par[1]
  g = alpha*exp(-alpha*x)
  G = 1- exp(-alpha*x)
  f = 2*g*exp(-(G/(1-G))^2)/(sqrt(pi)*(1-G)^2)
  f
}

# Fun??o de distribui??o da Erf_Exp
cdf_Erf_Exp <- function(par,x){
  alpha = par[1]
  G = 1- exp(-alpha*x)
  Fx = 2*pnorm((G/(1-G))*sqrt(2))-1
  Fx
}

# Log-verossimilhan?a da Erf_Exp
mlog_Erf_Exp <- function(par,x){
  s <- sum(log(pdf_Erf_Exp(par=par,x=x)))
  s
}



####################################################################
# Fun??o de densidade da Erf_Kuma
pdf_Erf_K <- function(par,x){
  alpha = par[1]
  beta = par[2]
  g = alpha*beta*x^(alpha-1)*(1-x^alpha)^(beta-1)
  G = 1-(1-x^alpha)^beta
  f = 2*g*exp(-(G/(1-G))^2)/(sqrt(pi)*(1-G)^2)
  f
}

# Fun??o de distribui??o da Erf_Kuma
cdf_Erf_K <- function(par,x){
  alpha = par[1]
  beta = par[2]
  g = alpha*beta*x^(alpha-1)*(1-x^alpha)^(beta-1)
  G = 1-(1-x^alpha)^beta
  Fx =  2*pnorm((G/(1-G))*sqrt(2))-1
  Fx
}

# Log-verossimilhan?a da Erf_Kuma
mlog_Erf_K <- function(par,x){
  s <- sum(log(pdf_Erf_K(par=par,x=x)))
  s
}




library(extraDistr)
####################################################################
# Fun??o de densidade da Erf_Gumbel
pdf_Erf_Gu <- function(par,x){
  alpha = par[1]
  beta = par[2]
  g = dgumbel(x, mu = alpha, sigma = beta)
  G = pgumbel(x, mu = alpha, sigma = beta)
  f = 2*g*exp(-(G/(1-G))^2)/(sqrt(pi)*(1-G)^2)
  f
}

# Fun??o de distribui??o da Erf_Gumbel
cdf_Erf_Gu <- function(par,x){
  alpha = par[1]
  beta = par[2]
  g = dgumbel(x, mu = alpha, sigma = beta)
  G = pgumbel(x, mu = alpha, sigma = beta)
  Fx =  2*pnorm((G/(1-G))*sqrt(2))-1
  Fx
}

# Log-verossimilhan?a da Erf_Gumbel
mlog_Erf_Gu <- function(par,x){
  s <- sum(log(pdf_Erf_Gu(par=par,x=x)))
  s
}




####################################################################
# Fun??o de densidade da Erf_Normal
pdf_Erf_N <- function(par,x){
  alpha = par[1]
  beta = par[2]
  g = dnorm(x, mean = alpha, sd = beta)
  G = pnorm(x, mean = alpha, sd = beta)
  f = 2*g*exp(-(G/(1-G))^2)/(sqrt(pi)*(1-G)^2)
  f
}

# Fun??o de distribui??o da Erf_Normal
cdf_Erf_N <- function(par,x){
  alpha = par[1]
  beta = par[2]
  g = dnorm(x, mean = alpha, sd = beta)
  G = pnorm(x, mean = alpha, sd = beta)
  Fx =  2*pnorm((G/(1-G))*sqrt(2))-1
  Fx
}

# Log-verossimilhan?a da Erf_Normal
mlog_Erf_N <- function(par,x){
  s <- sum(log(pdf_Erf_N(par=par,x=x)))
  s
}




####################################################################
# Fun??o de densidade da Erf_Gamma
pdf_Erf_Gama <- function(par,x){
  alpha = par[1]
  beta = par[2]
  g = dgamma(x, shape=alpha, rate = beta)
  G = pgamma(x, shape=alpha, rate = beta)
  f = 2*g*exp(-(G/(1-G))^2)/(sqrt(pi)*(1-G)^2)
  f
}

# Fun??o de distribui??o da Erf_Gama
cdf_Erf_Gama <- function(par,x){
  alpha = par[1]
  beta = par[2]
  g = dgamma(x, shape=alpha, rate = beta)
  G = pgamma(x, shape=alpha, rate = beta)
  Fx =  2*pnorm((G/(1-G))*sqrt(2))-1
  Fx
}

# Log-verossimilhan?a da Erf_Gama
mlog_Erf_Gama <- function(par,x){
  s <- sum(log(pdf_Erf_Gama(par=par,x=x)))
  s
}


####################################################################
# Fun??o de densidade da Erf_log-L
pdf_Erf_LL <- function(par,x){
  alpha = par[1]
  beta = par[2]
  g = (1+(x/alpha)^beta)^(-2)*(beta*x^(beta-1)/(alpha^beta))
  G = 1-(1+(x/alpha)^beta)^(-1)
  f = 2*g*exp(-(G/(1-G))^2)/(sqrt(pi)*(1-G)^2)
  f
}

# Fun??o de distribui??o da Erf_Log-L
cdf_Erf_LL <- function(par,x){
  alpha = par[1]
  beta = par[2]
  g = (1+(x/alpha)^beta)^(-2)*(beta*x^(beta-1)/(alpha^beta))
  G = 1-(1+(x/alpha)^beta)^(-1)
  Fx =  2*pnorm((G/(1-G))*sqrt(2))-1
  Fx
}

# Log-verossimilhan?a da Erf_Log-L
mlog_Erf_LL <- function(par,x){
  s <- sum(log(pdf_Erf_LL(par=par,x=x)))
  s
}


#####################################################################
# Log-verossimilhan?a da EGSGu
mlog_EGSGu <- function(par,x){
  s <- sum(log(pdf_EGSGu(par=par,x=x)))
  s
}

#####################################################################
# Log-verossimilhan?a da PrGumbel
mlog_egep <- function(par,x){
  sum(log(pdf_egep(par=par,x=x)))
}

#####################################################################
# Log-verossimilhan?a da PrW
mlog_ggp<- function(par,x){
  s <- sum(log(pdf_ggp(par=par,x=x)))
  s
}

#####################################################################

# Log-verossimilhan?a da KGP
mlog_kgp <- function(par,x){
  s <- sum(log(pdf_kumagp(par=par,x=x)))
  s
}
#####################################################################
# Log-verossimilhan?a da betagp
mlog_bgp <- function(par,x){
  s <- sum(log(pdf_betagp(par=par,x=x)))
  s
}


#####################################################################
# Log-verossimilhan?a da kumaBXII
mlog_kumaBXII<- function(par,x){
  s <- sum(log(pdf_kumaBXII(par=par,x=x)))
  s
}

#####################################################################
# Log-verossimilhan?a da BMW
mlog_BMW <- function(par,x){
  s <- sum(log(pdf_BMW(par=par,x=x)))
  s
}

#####################################################################
# Log-verossimilhan?a da EW
mlog_EW <- function(par,x){
  s <- sum(log(pdf_EW(par=par,x=x)))
  s
}

#####################################################################
# Log-verossimilhan?a da EE
mlog_EE <- function(par,x){
  s <- sum(log(pdf_EE(par=par,x=x)))
  s
}


#####################################################################
# Log-verossimilhan?a da KW
mlog_kw <- function(par,x){
  s <- sum(log(pdf_kw(par=par,x=x)))
  s
}
#####################################################################
# Log-verossimilhan?a da betaW
mlog_betaw <- function(par,x){
  s <- sum(log(pdf_betaw(par=par,x=x)))
  s
}

mlog_GW <- function(par,x){
  s <- sum(log(pdf_gammaw(par=par,x=x)))
  s
}


mlog_EGW <- function(par,x){
  s <- sum(log(pdf_EGW(par=par,x=x)))
  s
}

#####################################################################
# Log-verossimilhan?a da K-Gama
mlog_kgama <- function(par,x){
  s <- sum(log(pdf_kgama(par=par,x=x)))
  s
}

#####################################################################
# Log-verossimilhan?a da beta-Gama
mlog_betaG <- function(par,x){
  s <- sum(log(pdf_betaG(par=par,x=x)))
  s
}

#log-verossimilhan?a gama-gama
mlog_gamaG <- function(par,x){
  s <- sum(log(pdf_gammaG(par=par,x=x)))
  s
}

mlog_EGgama <- function(par,x){
  s <- sum(log(pdf_EGgama(par=par,x=x)))
  s
}

# Fun??o que calcula o AIC
# l ? a log-verossimilhan?a, q ? o n?mero de par?metros
AIc<- function(l,q){
  A <- -2*l+2*q
  A
}

# Fun??o que calcula o BIC
# l ? a log-verossimilhan?a, q ? o n?mero de par?metros
# n ? o tamanho da amostra 
BIc <- function(l,q,n){
  B <- -2*l+q*log(n)
  B
}

# Fun??o que calcula o CAIC
# l ? a log-verossimilhan?a, q ? o n?mero de par?metros
# n ? o tamanho da amostra
CAIC <- function(l,q,n){
  C <- -2*l+2*q*n/(n-q-1)
  C
}


# Fun??o que calcula a estat?stica de Cr?mer Von Mises para a
# distribui??o EGEP, theta.ch1 ? o vetor de par?metros estimados
# por mv para a distribui??o GGP

cramer <- function(theta,x,n,fda){
  x1 = sort(x)
  v = fda(par=theta,x=x1)
  y = qnorm(v)
  u = pnorm((y-mean(y))/sd(y))
  aux=rep(0,n)
  for(i in 1:n){
    aux[i]=(u[i]-(2*i-1)/(2*n))^2
  }
  W2=sum(aux)+1/(12*n)
  W2*(1+0.5/n)
}

# Fun??o que calcula a estat?stica de Anderson Darling para a
# distribui??o EGEP, theta.ch1 ? o vetor de par?metros estimados
# por mv para a distribui??o GGP


AD <- function(theta,x,n,fda){
  x1 = sort(x)
  v = fda(par=theta,x=x1)
  y = qnorm(v)
  u = pnorm((y-mean(y))/sd(y))
  aux=rep(0,n)
  for(i in 1:n){
    aux[i]=(2*i-1)*log(u[i])+(2*n+1-2*i)*log(1-u[i])
  }
  A2 = -n -(1/n)*sum(aux)
  A2*(1+0.75/n+2.25/n^2)
}


# Fun??o que calcula a estat?stica de Kolmogorov-Smirnov para a
# distribui??o GGP, theta.ch1 ? o vetor de par?metros estimados
# por mv para a distribui??o GGP


KS <- function(theta,x,n,fda){
  y = sort(x)
  aux1=rep(0,n)
  aux2=rep(0,n)
  for(i in 1:n){
    aux1[i] = fda(par=theta,x=y[i])-(i-1)/n
    aux2[i] = i/n-fda(par=theta,x=y[i])
  }   
  max(max(aux1),max(aux2),0)
}


est <- function(M,fda){
  out <- M$estimate
  l <- logLik(M)
  p <- nrow(out)
  a <- AIc(l,p)
  b <- BIc(l,p,n=n)
  c <- CAIC(l,p,n=n)
  ad <- AD(out[,1],x=x,n=n,fda=fda)
  ks <- KS(out[,1],x=x,n=n,fda=fda)
  cr <- cramer(out[,1],x=x,n=n,fda=fda)
  s <- cbind(cr,ks,ad,a,b,c)
  colnames(s)=c("Cramer","KS","AD","AIC","BIC","CAIC")
  round(s,4)
}

par <- function(M){
  out <- M$estimate
  out[,1]
}


ep <- function(M){
  out <- M$estimate
  out[,2]
}


#-----------------------------------------------------------------
#                            DADOS
#-----------------------------------------------------------------

#x <- c(1.43, 0.11, 0.71, 0.77, 2.63, 1.49, 3.46, 2.46, 0.59, 0.74, 1.23, 0.94, 4.36, 0.40, 1.74, 4.73, 2.23, 0.45, 0.70, 1.06, 1.46, 0.30, 1.82, 2.37, 0.63, 1.23, 1.24, 1.97, 1.86, 1.17)

#dados air conditionated
x <- c( 194, 413, 90, 74, 55, 23, 97, 50, 359, 50, 130, 487, 102, 15, 14, 10, 57, 320, 261, 51, 44, 9 , 254, 493, 18, 209, 41, 58, 60, 48, 56, 87, 11, 102, 12, 5, 100, 14, 29, 37, 186, 29, 104, 7, 4, 72, 270, 283 , 7 , 57, 33, 100, 61, 502, 220, 120, 141, 22, 603, 35, 98, 54, 181, 65, 49, 12, 239, 14, 18, 39, 3, 12, 5, 32, 9, 14, 70, 47, 62, 142, 3, 104, 85, 67, 169, 24, 21, 246, 47, 68, 15, 2, 91, 59, 447, 56, 29, 176, 225, 77, 197, 438, 43, 134, 184, 20, 386, 182, 71, 80, 188, 230, 152, 36, 79, 59, 33, 246, 1, 79, 3, 27, 201, 84, 27, 21, 16, 88, 130, 14, 118, 44, 15, 42, 106, 46, 230, 59, 153, 104, 20, 206, 5 , 66, 34, 29, 26, 35, 5, 82, 5, 61, 31, 118, 326, 12, 54, 36, 34, 18, 25, 120, 31, 22, 18, 156, 11, 216, 139, 67, 310, 3, 46, 210, 57, 76, 14, 111, 97, 62, 26, 71, 39, 30, 7, 44, 11, 63, 23, 22, 23, 14, 18, 13, 34, 62, 11, 191, 14, 16, 18, 130, 90, 163, 208, 1, 24, 70, 16, 101, 52, 208 , 95 )

#x <- c(0.22 , 0.17 , 0.11 , 0.10 , 0.15 , 0.06 , 0.05 , 0.07 , 0.12 , 0.09 , 0.23 , 0.25 , 0.23,		
#       0.24 , 0.20 , 0.08 , 0.11 , 0.12 , 0.10 , 0.06 , 0.20 , 0.17 , 0.20 , 0.11 , 0.16 , 0.09,		
#       0.10 , 0.12 , 0.12 , 0.10 , 0.09 , 0.17 , 0.19 , 0.21 , 0.18 , 0.26 , 0.19 , 0.17 , 0.18,		
#       0.20 , 0.24 , 0.19 , 0.21 , 0.22 , 0.17 , 0.08 , 0.08 , 0.06 , 0.09 , 0.22 , 0.23 , 0.22,		
#       0.19 , 0.27 , 0.16 , 0.28 , 0.11 , 0.10 , 0.20 , 0.12 , 0.15 , 0.08 , 0.12 , 0.09 , 0.14,		
#       0.07 , 0.09 , 0.05 , 0.06 , 0.11 , 0.16 , 0.20 , 0.25 , 0.16 , 0.13 , 0.11 , 0.11 , 0.11,		
#       0.08 , 0.22 , 0.11 , 0.13 , 0.12 , 0.15 , 0.12 , 0.11 , 0.11 , 0.15 , 0.10 , 0.15 , 0.17,		
#       0.14 , 0.12 , 0.18 , 0.14 , 0.18 , 0.13 , 0.12 , 0.14 , 0.09 , 0.10 , 0.13 , 0.09 , 0.11,		
#       0.11 , 0.14 , 0.07 , 0.07 , 0.19 , 0.17 , 0.18 , 0.16 , 0.19 , 0.15 , 0.07 , 0.09 , 0.17,		
#       0.10 , 0.08 , 0.15 , 0.21 , 0.16 , 0.08 , 0.10 , 0.06 , 0.08 , 0.12 , 0.13)

# Tamanho da amostra
n=length(x)

par_egep <- c(0.0004,0.002,2)
#par_egep <- c(0.011,0.025,1.616)
maxi_egep <- summary(maxLik(mlog_egep, start=par_egep,method= "SANN",x=x))
maxi_egep


# GGP
par_ggp <- c(0.2,9.5,5.5)
maxi_ggp <- summary(maxLik(mlog_ggp, start=par_ggp,method= "SANN",x=x))
maxi_ggp

# bgp
par_bgp <- c(0.005,0.01,1.9,16.6)
maxi_bgp <- summary(maxLik(mlog_bgp, start=par_bgp,method= "SANN",x=x))
maxi_bgp

# kgp
#par_kgp <- c(2.5,0.1,5.9,25.6)
par_kgp <- c(4.5,1.01,5,35.6)
maxi_kgp <- summary(maxLik(mlog_kgp, start=par_kgp,method= "SANN",x=x))
maxi_kgp

#EGSGu
par_EGSGu <- c(5.0005,24)
maxi_EGSGu <- summary(maxLik(mlog_EGSGu, start=par_EGSGu,method= "SANN", x=x))
maxi_EGSGu

# EW
par_EW <- c(3.5,2,2)
maxi_EW <- summary(maxLik(mlog_EW, start=par_EW,method= "SANN",x=x))
maxi_EW

# BMW
par_BMW <- c(3.344, 0.942,1.02,10.883,1.617)
maxi_BMW <- summary(maxLik(mlog_BMW, start=par_BMW,method= "SANN",x=x))
maxi_BMW

# kumaBXII
#par_kumaBXII <- c(11.413, 7,1.3,0.883,0.617)
par_kumaBXII <- c(16.4,6.5,3.9,0.07,2.1)
maxi_kumaBXII <- summary(maxLik(mlog_kumaBXII, start=par_kumaBXII,method= "SANN",x=x))
maxi_kumaBXII

# EE
par_EE <- c(2,2)
maxi_EE <- summary(maxLik(mlog_EE, start=par_EE,method= "SANN",x=x))
maxi_EE

#Kuma-Weibull
par_kw <- c(0.7,15.1,1.3,0.1)
maxi_kw <- summary(maxLik(mlog_kw, start=par_kw,method= "SANN",x=x))
maxi_kw

#Beta-weibull
par_betaw <- c(0.55,15.55,1.6,1.6)
maxi_betaw <- summary(maxLik(mlog_betaw, start=par_betaw,method= "SANN",x=x))
maxi_betaw

# Gamma-Weibull
par_gammaw <- c(4,0.5,2.6)
maxi_gammaw <- summary(maxLik(mlog_GW, start=par_gammaw,method= "SANN",x=x))
maxi_gammaw

##EG-Weibull
par_EGW <- c(0.004,2.022,7.4,0.005)
maxi_EGW <- summary(maxLik(mlog_EGW, start=par_EGW,method= "SANN",x=x))
maxi_EGW

# kuma-Gama
par_kgama <- c(43,  0.10765378, 2.26725909,25.23827009)
maxi_kgama <- summary(maxLik(mlog_kgama, start=par_kgama,method= "SANN",x=x))
maxi_kgama

# beta-gama
par_betaG <- c(1.250571,1.5,5.1147976,9.4187521)
maxi_betaG <- summary(maxLik(mlog_betaG, start=par_betaG,method= "SANN",x=x))
maxi_betaG

# Gamma-Gama
par_gammaG <- c(13,6.5,20.6)
maxi_gammaG <- summary(maxLik(mlog_gamaG, start=par_gammaG,method= "SANN",x=x))
maxi_gammaG

##EG-Gama
par_EGgama <- c(1.2994783, 1.15070248,   6.8546916, 2.05386948)
maxi_EGgama <- summary(maxLik(mlog_EGgama, start=par_EGgama,method= "SANN",x=x))
maxi_EGgama

#Erf-We
par_Erf_We <- c(0.00001,1.5)
maxi_Erf_We <- summary(maxLik(mlog_Erf_We, start=par_Erf_We,method= "SANN",x=x))
maxi_Erf_We

#Erf_Exp
par_Erf_Exp <- c(0.000413)
maxi_Erf_Exp <- summary(maxLik(mlog_Erf_Exp, start=par_Erf_Exp,method= "SANN",x=x))
maxi_Erf_Exp

#Erf_Kuma
par_Erf_K <- c(55,39)
maxi_Erf_K <- summary(maxLik(mlog_Erf_K, start=par_Erf_K,method= "SANN",x=x))
maxi_Erf_K

#Erf_Gumbel
par_Erf_Gu <- c(90,120)
maxi_Erf_Gu <- summary(maxLik(mlog_Erf_Gu, start=par_Erf_Gu,method= "SANN",x=x))
maxi_Erf_Gu

#Erf_Normal
par_Erf_N <- c(9,50)
maxi_Erf_N <- summary(maxLik(mlog_Erf_N, start=par_Erf_N,method= "SANN",x=x))
maxi_Erf_N

#Erf_Gamma
par_Erf_Gama <- c(10,0.1)
maxi_Erf_Gama <- summary(maxLik(mlog_Erf_Gama, start=par_Erf_Gama,method= "SANN",x=x))
maxi_Erf_Gama

#Erf_Log_Logistic
par_Erf_LL <- c(0.1,0.1)
maxi_Erf_LL <- summary(maxLik(mlog_Erf_LL, start=par_Erf_LL,method= "SANN",x=x))
maxi_Erf_LL


EST <- rbind(est(maxi_egep,cdf_egep),est(maxi_ggp,cdf_ggp),est(maxi_bgp,cdf_bgp),est(maxi_kgp,cdf_kgp),est(maxi_EGSGu,cdf_EGSGu),
             est(maxi_EW,cdf_EW),est(maxi_BMW,cdf_BMW),est(maxi_kumaBXII,cdf_kumaBXII),est(maxi_EE,cdf_EE),
             est(maxi_kw,cdf_kw),est(maxi_betaw,cdf_betaw),est(maxi_gammaw,cdf_gammaw),est(maxi_EGW,cdf_EGW),
             est(maxi_kgama,cdf_kgama),est(maxi_betaG,cdf_betaG),est(maxi_gammaG,cdf_gammaG),est(maxi_EGgama,cdf_EGgama),
             est(maxi_Erf_We,cdf_Erf_We),est(maxi_Erf_Exp,cdf_Erf_Exp),est(maxi_Erf_K,cdf_Erf_K),
             est(maxi_Erf_Gu,cdf_Erf_Gu),est(maxi_Erf_N,cdf_Erf_N),est(maxi_Erf_Gama,cdf_Erf_Gama),
             est(maxi_Erf_LL,cdf_Erf_LL))


EST <- rbind(est(maxi_egep,cdf_egep),est(maxi_ggp,cdf_ggp),est(maxi_bgp,cdf_bgp),
             est(maxi_kgp,cdf_kgp),
             est(maxi_EW,cdf_EW),est(maxi_kumaBXII,cdf_kumaBXII),est(maxi_EE,cdf_EE),
             est(maxi_kw,cdf_kw),est(maxi_betaw,cdf_betaw),est(maxi_gammaw,cdf_gammaw),
             est(maxi_gammaG,cdf_gammaG),
             est(maxi_Erf_We,cdf_Erf_We),est(maxi_Erf_Exp,cdf_Erf_Exp),
             est(maxi_Erf_Gama,cdf_Erf_Gama),
             est(maxi_Erf_LL,cdf_Erf_LL))

rownames(EST)=c("EGEP","GGP","BGP","KGP","EW","kumaBXII","EE","Kuma-W","Beta-W","Gama-W","Gama-Gama","Erf-We","Erf-Exp","Erf-Gama","Erf-LL")


PAR <- rbind(par(maxi_egep),par(maxi_ggp),par(maxi_bgp),par(maxi_kgp),par(maxi_EGSGu),par(maxi_EW),par(maxi_BMW),par(maxi_kumaBXII),
             par(maxi_EE),par(maxi_kw),par(maxi_betaw),par(maxi_gammaw),par(maxi_EGW),
             par(maxi_kgama),par(maxi_betaG),par(maxi_gammaG),par(maxi_EGgama),par(maxi_Erf_We,cdf_Erf_We),par(maxi_Erf_Exp,cdf_Erf_Exp),
             par(maxi_Erf_K,cdf_Erf_K),par(maxi_Erf_Gu,cdf_Erf_Gu),par(maxi_Erf_N,cdf_Erf_N),par(maxi_Erf_Gama,cdf_Erf_Gama),
             par(maxi_Erf_LL,cdf_Erf_LL))

EP <- rbind(ep(maxi_egep),ep(maxi_ggp),ep(maxi_bgp),ep(maxi_kgp),ep(maxi_EGSGu),ep(maxi_EW),ep(maxi_BMW),ep(maxi_kumaBXII),
            ep(maxi_EE),ep(maxi_kw),ep(maxi_betaw),ep(maxi_gammaw),ep(maxi_EGW),
            ep(maxi_kgama),ep(maxi_betaG),ep(maxi_gammaG),ep(maxi_EGgama),ep(maxi_Erf_We,cdf_Erf_We),ep(maxi_Erf_Exp,cdf_Erf_Exp),
            ep(maxi_Erf_K,cdf_Erf_K),ep(maxi_Erf_Gu,cdf_Erf_Gu),ep(maxi_Erf_N,cdf_Erf_N),ep(maxi_Erf_Gama,cdf_Erf_Gama),
            ep(maxi_Erf_LL,cdf_Erf_LL))


rownames(EST)=c("EGEP","GGP","BGP","KGP","EGSGu","EW","BMW","kumaBXII","EE","Kuma-W","Beta-W","Gama-W","EGW","Kuma-Gama","Beta-Gama","Gama-Gama","EG-Gama","Erf-We","Erf-Exp","Erf-K","Erf-Gu","Erf-N","Erf-Gama","Erf-LL")
rownames(PAR)=c("EGEP","GGP","BGP","KGP","EGSGu","EW","BMW","kumaBXII","EE","Kuma-W","Beta-W","Gama-W","EGW","Kuma-Gama","Beta-Gama","Gama-Gama","EG-Gama","Erf-We","Erf-Exp","Erf-K","Erf-Gu","Erf-N","Erf-Gama","Erf-LL")
rownames(EP)=c("EGEP","GGP","BGP","KGP","EGSGu","EW","BMW","kumaBXII","EE","Kuma-W","Beta-W","Gama-W","EGW","Kuma-Gama","Beta-Gama","Gama-Gama","EG-Gama","Erf-We","Erf-Exp","Erf-K","Erf-Gu","Erf-N","Erf-Gama","Erf-LL")

round(EST,3)
round(PAR,3)
round(EP,3)







# Este ? o script para fazer o gr?fico da densidade
xx=seq(0.001,5,by=0.01)
hist(x,breaks=c(0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5),ylab="Density",xlab="Time between failures",main="",prob=T)
lines(xx,pdf_EGSGu(out5[,1],x=xx),type="l",lwd=2,lty=2)
lines(xx,pdf_ggp(out2[,1],x=xx),type="l",lwd=2,lty=2,col=2)
lines(xx,pdf_egep(out1[,1],x=xx),type="l",lwd=2,lty=2,col=3)
lines(xx,pdf_kumagp(out4[,1],x=xx),type="l",lwd=2,lty=2,col=3)
lines(xx,pdf_betagp(out3[,1],x=xx),type="l",lwd=2,lty=2,col=4)
lines(xx,pdf_kumaBXII(out8[,1],x=xx),type="l",lwd=2,lty=2,col=5)
lines(xx,pdf_BMW(out7[,1],x=xx),type="l",lwd=2,lty=2,col=6)
lines(xx,pdf_EW(out6[,1],x=xx),type="l",lwd=2,lty=2,col=7)
lines(xx,pdf_EE(out9[,1],x=xx),type="l",lwd=2,lty=2,col=8)
legend(3,0.55,legend=c("pdf EGSGu"),lwd = c(2), lty = c(2),bty="n")



# Este ? o script para fazer o gr?fico da distribui??o
xx=seq(0.001,5,by=0.01)
P=ecdf(x)
plot(P,verticals = TRUE,do.points = FALSE,ylab="CDF",xlab="Time between failures",main="",lwd=2,ylim=(c(0,1)))
lines(xx,cdf_EGSGu(out5[,1],x=xx),type="l",lwd=2,lty=2)
lines(xx,cdf_ggp(out2[,1],x=xx),type="l",lwd=2,lty=2,col=2)
lines(xx,cdf_egep(out1[,1],x=xx),type="l",lwd=2,lty=2,col=3)
lines(xx,cdf_kumagp(out4[,1],x=xx),type="l",lwd=2,lty=2,col=3)
lines(xx,cdf_betagp(out3[,1],x=xx),type="l",lwd=2,lty=2,col=4)
lines(xx,cdf_kumaBXII(out8[,1],x=xx),type="l",lwd=2,lty=2,col=5)
lines(xx,cdf_BMW(out7[,1],x=xx),type="l",lwd=2,lty=2,col=6)
lines(xx,cdf_EW(out6[,1],x=xx),type="l",lwd=2,lty=2,col=7)
lines(xx,cdf_EE(out9[,1],x=xx),type="l",lwd=2,lty=2,col=8)
legend(3,0.55,legend=c("cdf EGSGu"),lwd = c(2), lty = c(2),bty="n")
