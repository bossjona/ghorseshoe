#' Grouped horseshoe regression Gibbs sampler.
#'
#' @param X A (n x M) matrix of covariates that we want to apply grouped horseshoe shrinkage on.
#' @param C A (n x K) matrix of covariates that we want to apply ridge regression shrinkage on (typically adjustment covariates).
#' @param Y A (n x 1) column vector of responses.
#' @param grp_idx A (1 x M) row vector indicating which group of the J groups the p covariates in X belong to.
#' @param alpha_inits A (K x 1) column vector containing initial values for the regression coefficients corresponding to C.
#' @param beta_inits A (M x 1) column vector containing initial values for the regression coefficients corresponding to X.
#' @param lambda_sq_inits A (M x 1) column vector containing initial values for the local shrinkage parameters.
#' @param gamma_sq_inits A (J x 1) column vector containing initial values for the group shrinkage parameters.
#' @param eta_inits A (J x 1) column vector containing initial values for the mixing parameters.
#' @param mu_init Initial value for the intercept parameter (double).
#' @param tau_sq_init Initial value for the global shrinkage parameter (double).
#' @param sigma_sq_init Initial value for the residual variance (double).
#' @param psi_sq_init Initial value for the variailbity in the alphas (double).
#' @param nu_init Initial value for the augmentation variable (double).
#' @param n_burn_in The number of burn-in samples (integer).
#' @param n_samples The number of posterior draws (integer).
#' @param n_thin The thinning interval (integer).
#' @param error_tol Parameter that controls numerical stability of the algorithm (double).
#' @param a1 Shape parameter for the inverse gamma hyperprior on psi_sq (double).
#' @param b1 Scale parameter for the inverse gamma hyperprior on psi_sq (double).
#' @param a2 Shape parameter for the gamma hyperprior on the etas (double).
#' @param b2 Scale parameter for the gamma hyperprior on the etas (double).
#' @param verbose Boolean value which indicates whether or not to print the progress of the Gibbs sampler.
#' @return A list containing the posterior draws of (1) the intercept term (mus) (2) the regression coefficients (alphas and betas) (3) the individual shrinkage parameters (lambda_sqs) (4) the group shrinkage parameters (gamma_sqs) (5) the global shrinkage parameter (tau_sqs) (6) the residual error variance (sigma_sqs) (7) the mixing parameter (etas) and (8) the variance of the prior on alpha (psi_sqs). The list also contains the specified hyperparameters (hyperparam_a1, hyperparam_b1, hyperparam_a2, and hyperparam_b2), details regarding the dataset (X, C, Y, grp_idx), and Gibbs sampler details (n_burn_in, n_samples, and n_thin).
grouped_horseshoe <- function(X, C, Y, grp_idx, alpha_inits = rep(0, ncol(C)), beta_inits = rep(0, ncol(X)), lambda_sq_inits = rep(1, ncol(X)), gamma_sq_inits = rep(1.0, length(unique(grp_idx))), eta_inits = rep(1.0, length(unique(grp_idx))), mu_init = 0, tau_sq_init = 1, sigma_sq_init = 1, psi_sq_init = 1,
                              nu_init = 1, n_burn_in = 500, n_samples = 1000, n_thin = 1, error_tol = 1e-07, a1 = 0.001, b1 = 0.001, a2 = 0.001, b2 = 0.001, verbose = TRUE){
  #Store useful quantites
  grp_size <- as.vector(table(grp_idx))
  grp_size_cs <- cumsum(grp_size)
  
  #Fit grouped horseshoe
  gh <- grouped_horseshoe_gibbs_sampler(X = X, C = C, Y = Y, grp_idx = grp_idx, grp_size = grp_size, grp_size_cs = grp_size_cs,
                                          alpha_inits = alpha_inits, beta_inits = beta_inits, lambda_sq_inits = lambda_sq_inits, gamma_sq_inits = gamma_sq_inits,
                                          eta_inits = eta_inits, mu_init = mu_init, tau_sq_init = tau_sq_init, sigma_sq_init = sigma_sq_init, psi_sq_init = psi_sq_init,
                                          nu_init = nu_init, n_burn_in = n_burn_in, n_samples = n_samples, n_thin = n_thin, error_tol = error_tol,
                                          a1 = a1, b1 = b1, a2 = a2, b2 = b2, verbose = verbose)
  return(gh)
}
