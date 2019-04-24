library(MASS)
library(devtools)

#Install ghorseshoe package
install_github("https://github.com/bossjona/ghorseshoe")
library(ghorseshoe)

#Function that generates simulated data
sim.multipollutant.data <- function(n, rho, grps, grp.sizes, seed, resid.var, alpha, beta, mu){
  K <- length(alpha)
  M <- length(beta)
  set.seed(seed)
  submats <- list()
  idx <- c(0,cumsum(grp.sizes))
  for(cor in 1:length(rho)){
    submats[[cor]] <- diag(1, grp.sizes[cor])
    submats[[cor]][submats[[cor]] == 0] <- rho[cor]
  }
  Sig <- matrix(data = rep(0,M^2), ncol = M, nrow = M)
  for(g in 1:length(grp.sizes)){
    Sig[(idx[g]+1):idx[g+1],(idx[g]+1):idx[g+1]] <- submats[[g]]
  }
  X <- mvrnorm(n=n, mu = rep(0,M), Sigma = Sig)
  C <- mvrnorm(n=n, mu = rep(0,K), Sigma = diag(1,K))
  Y <- mu + C%*%alpha + X%*%beta + rnorm(n, mean = 0, sd = sqrt(resid.var))
  return(list(X = X, C = C, Y = Y, Sig = Sig, M = M, K = K, n = n, rho = rho, grps = grps, grp.sizes = grp.sizes,
              seed = seed, resid.var = resid.var, mu = mu, alpha = alpha, beta = beta))
}

#Simulated dataset parameters
n <- 1000
rho <- c(0.8,0.8,0.1,0.1)
grps <- c(rep(1,15),rep(2,15),rep(3,15),rep(4,15))
grp.sizes <- rep(15,4)
seed <- 20192803
resid.var <- 100
alpha <- rep(1,5)
beta <- c(rep(0,15), c(rep(1,12), rep(0,3)), rep(0,15), c(rep(1,3), rep(0,12)))
mu <- 0

#Simulated data
gen.data <- sim.multipollutant.data(n = n, rho = rho, grps = grps, grp.sizes = grp.sizes, seed = seed,
                                    resid.var = resid.var, alpha = alpha, beta = beta, mu = mu)

#Grouped horseshoe regression example
ghorseshoe.fit <- grouped_horseshoe(X = gen.data$X, C = gen.data$C, Y = gen.data$Y, grp_idx = gen.data$grps,
                                        alpha_inits = rep(0.0, ncol(gen.data$C)), beta_inits = rep(0.0, ncol(gen.data$X)),
                                        lambda_sq_inits = rep(1.0, ncol(gen.data$X)), gamma_sq_inits = rep(1.0, length(unique(gen.data$grps))),
                                        eta_inits = rep(1.0,length(unique(gen.data$grps))), n_samples = 2000, n_thin = 5, n_burn_in = 1000, verbose = TRUE)

