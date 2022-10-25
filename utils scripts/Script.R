library(erfG)

AIc<- function(l,q){
  A <- -2*l+2*q
  A
}

BIc <- function(l,q,n){
  B <- -2*l+q*log(n)
  B
}

CAIC <- function(l,q,n){
  C <- -2*l+2*q*n/(n-q-1)
  C
}


log_lerf <- function(x, dist, param) {
  param = param
  prob = vector(length=length(x))
  for(j in x){
    call = paste0('erfG:::d',dist, '(', j)
    for(i in param){
      call = paste0(call,',',i)
    }
    call = paste0(call, ')')
    prob[match(j,x)] = parse(text=call) |> eval()
  }
  return(sum(log(prob)))
}


pdf_EE <- function(par,x){
  alpha = par[1]
  lambda = par[2]
  sum(log(alpha*lambda*exp(-lambda*x)*(1-exp(-lambda*x))^(alpha-1)))
}


Wei <- function(par,x){
  shape = par[1]
  scale = par[2]
  sum(log(dweibull(x, shape = shape, scale = scale)))
}


norm  <- function(par,x){
  mean = par[1]
  sd = par[2]
  sum(log(dnorm(x, mean = mean, sd = sd)))
}


est <- function(M, n){
  out <- M$estimate
  l <- logLik(M)
  p <- nrow(out)
  a <- AIc(l,p)
  b <- BIc(l,p,n=n)
  c <- CAIC(l,p,n=n)
  s <- cbind(a,b,c)
  colnames(s)=c("AIC","BIC","CAIC")
  round(s,4)
}

# Variável:

x = survival::veteran$time


p1 <- quantile(x,p=.632)
ws <- sort(x)
Fh <- ppoints(ws)
k0 <- lm(log(-log(1-Fh))~log(ws))$coefficients[2]

# Chutes iniciais:

par = c(k0, p1/10)
par = c(mean(x), sd(x))

(s1 = summary(maxLik::maxLik(log_lerf,start=par, x=x, dist='weibull')))
(s1 = summary(maxLik::maxLik(pdf_Erf_We,start=par, x=x)))
(s2 = summary(maxLik::maxLik(Wei,start=par, x=x)))
s3 = summary(maxLik::maxLik(norm,start=par, x=x))
s4 = summary(maxLik::maxLik(pdf_EE,start=par, x=x))

rbind(est(s1, length(x)),est(s2, length(x)),est(s3, length(x)),est(s4, length(x)))
