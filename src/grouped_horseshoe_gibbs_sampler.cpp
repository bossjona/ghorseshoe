#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

//' Function that calls the rgig function from the GeneralizedHyperbolic package in R.
//'
//' Randomly generates one draw from a generalized inverse gaussian distribution.
//' @param chi A positive double.
//' @param psi A positive double.
//' @param lambda A non-negative double.
//' @return A random draw from the generalized inverse gaussian distribution with parameters chi, psi, and lambda (double).
// [[Rcpp::export]]
double rgig_use(double chi, double psi, double lambda) {
	// Obtaining namespace of GeneralizedHyperbolic package
	Environment pkg = Environment::namespace_env("GeneralizedHyperbolic");

	// calling rgig() from GeneralizedHyperbolic package
	Function f = pkg["rgig"];
	SEXP draw = f(Named("n",1), Named("chi", chi), Named("psi", psi), Named("lambda",lambda));
	double ret = Rcpp::as<double>(draw);
	return ret;
}

//' Gibbs sampler for the grouped horseshoe model.
//'
//' An Rcpp function that implements a Gibbs sampler for the grouped horseshoe model.
//' @param X A (n x M) matrix of covariates that we want to apply grouped horseshoe shrinkage on.
//' @param C A (n x K) matrix of covariates that we want to apply ridge regression shrinkage on (typically adjustment covariates).
//' @param Y A (n x 1) column vector of responses.
//' @param grp_idx A (1 x M) row vector indicating which group of the J groups the p covariates in X belong to.
//' @param grp_size A (1 x J) row vector indicating the number of covariates in each group.
//' @param grp_size_cs A (1 x J) row vector that is the cumulative sum of grp_size (indicating the indicies where each group ends).
//' @param alpha_inits A (K x 1) column vector containing initial values for the regression coefficients corresponding to C.
//' @param beta_inits A (M x 1) column vector containing initial values for the regression coefficients corresponding to X.
//' @param lambda_sq_inits A (M x 1) column vector containing initial values for the local shrinkage parameters.
//' @param gamma_sq_inits A (J x 1) column vector containing initial values for the group shrinkage parameters.
//' @param eta_inits A (J x 1) column vector containing initial values for the mixing parameters.
//' @param mu_init Initial value for the intercept parameter (double).
//' @param tau_sq_init Initial value for the global shrinkage parameter (double).
//' @param sigma_sq_init Initial value for the residual variance (double).
//' @param psi_sq_init Initial value for the variailbity in the alphas (double).
//' @param nu_init Initial value for the augmentation variable (double).
//' @param n_burn_in The number of burn-in samples (integer).
//' @param n_samples The number of posterior draws (integer).
//' @param n_thin The thinning interval (integer).
//' @param error_tol Parameter that controls numerical stability of the algorithm (double).
//' @param a1 Shape parameter for the inverse gamma hyperprior on psi_sq (double).
//' @param b1 Scale parameter for the inverse gamma hyperprior on psi_sq (double).
//' @param a2 Shape parameter for the gamma hyperprior on the etas (double).
//' @param b2 Scale parameter for the gamma hyperprior on the etas (double).
//' @param verbose Boolean value which indicates whether or not to print the progress of the Gibbs sampler.
//' @return A list containing the posterior draws of (1) the intercept term (mus) (2) the regression coefficients (alphas and betas) (3) the individual shrinkage parameters (lambda_sqs) (4) the group shrinkage parameters (gamma_sqs) (5) the global shrinkage parameter (tau_sqs) (6) the residual error variance (sigma_sqs) (7) the mixing parameter (etas) and (8) the variance of the prior on alpha (psi_sqs). The list also contains the specified hyperparameters (hyperparam_a1, hyperparam_b1, hyperparam_a2, and hyperparam_b2), details regarding the dataset (X, C, Y, grp_idx), and Gibbs sampler details (n_burn_in, n_samples, and n_thin).
// [[Rcpp::export]]
List grouped_horseshoe_gibbs_sampler(arma::mat& X, arma::mat& C, arma::colvec& Y, arma::rowvec& grp_idx, arma::rowvec& grp_size, arma::rowvec& grp_size_cs,
	arma::colvec& alpha_inits, arma::colvec& beta_inits, arma::colvec& lambda_sq_inits, arma::colvec& gamma_sq_inits, arma::colvec& eta_inits, double mu_init = 0, double tau_sq_init = 1, double sigma_sq_init = 1, double psi_sq_init = 1,
	double nu_init = 1, int n_burn_in = 500, int n_samples = 1000, int n_thin = 1, double error_tol = 1e-07, double a1 = 0.001, double b1 = 0.001, double a2 = 0.001, double b2 = 0.001, bool verbose = true) {

	//Pre-compute and store useful quantities
	int n = X.n_rows;
	int K = C.n_cols;
	int J = eta_inits.n_elem;
	int M = X.n_cols;
	arma::mat tX = X.t();
	arma::mat tC = C.t();
	arma::mat XtX = tX * X;
	arma::mat CtC = tC * C;
	arma::colvec ones_vec = arma::ones(n);
	arma::mat diagK = diagmat(arma::ones(K));

	//Initialize
	arma::colvec alpha = alpha_inits;
	arma::colvec beta = beta_inits;
	arma::colvec lambda_sq = lambda_sq_inits;
	arma::colvec gamma_sq = gamma_sq_inits;
	arma::colvec eta = eta_inits;
	double mu = mu_init;
	double tau_sq = tau_sq_init;
	double sigma_sq = sigma_sq_init;
	double psi_sq = psi_sq_init;
	double nu = nu_init;

	//Store Gibbs sampler output
	arma::colvec mu_store = arma::zeros(n_samples);
	arma::mat alpha_store = arma::zeros(n_samples, K);
	arma::mat beta_store = arma::zeros(n_samples, M);
	arma::mat lambda_store = arma::zeros(n_samples, M);
	arma::mat gamma_store = arma::zeros(n_samples, J);
	arma::colvec tau_store = arma::zeros(n_samples);
	arma::colvec sigma_store = arma::zeros(n_samples);
	arma::mat eta_store = arma::zeros(n_samples, J);
	arma::colvec psi_store = arma::zeros(n_samples);
	arma::colvec nu_store = arma::zeros(n_samples);

	//Prevent repetative initializations by initializing here
	arma::mat alpha_tmp = arma::zeros(K, K);
	arma::mat local_param_inv = arma::zeros(M, M);
	arma::colvec local_param_expand_inv = arma::zeros(M);
	arma::mat beta_tmp = arma::zeros(M, M);
	double stable_psi = 0;
	double sum_inv_lambda_sq = 0;

	int cnt = 0;
	while (cnt < n_burn_in) {

		//Draw mu
		mu = R::rnorm(sum(Y - C * alpha - X * beta) / (double)n, sqrt(sigma_sq/(double)n));

		//Draw alpha
		alpha_tmp = inv((1.0 / sigma_sq)*CtC + (1.0 / psi_sq)*diagK);
		alpha = arma::mvnrnd((1.0 / sigma_sq) * alpha_tmp * tC * (Y - mu * ones_vec - X * beta), alpha_tmp);

		//Draw beta
		for (int g = 0; g < M; ++g) {
			local_param_expand_inv[g] = 1.0 / (gamma_sq[grp_idx[g] - 1] * lambda_sq[g]);
		}
		local_param_inv.diag() = local_param_expand_inv;
		beta_tmp = inv((1.0 / sigma_sq) * XtX + (1.0 / tau_sq) * local_param_inv);
		beta = arma::mvnrnd((1.0 / sigma_sq) * beta_tmp * tX * (Y - mu * ones_vec - C * alpha), beta_tmp);

		//Draw tau^2
		tau_sq = 1.0 / R::rgamma(((double)M + 1.0) / 2.0, 1.0 / ((beta.t() * local_param_inv * beta) / 2.0 + 1.0 / nu).eval()(0, 0));
		
		//Draw sigma^2
		sigma_sq = 1.0 / R::rgamma(((double)n + 1.0) / 2.0, 1.0 / (((Y - mu * ones_vec - C * alpha - X * beta).t()*(Y - mu * ones_vec - C * alpha - X * beta)) / 2.0 + 1.0 / nu).eval()(0, 0));

		for (int j = 0; j < J; ++j) {
			//Draw gamma^2
			if (j != 0) {
				stable_psi = 0;
				for (int l = grp_size_cs[j - 1]; l < grp_size_cs[j]; ++l) {
					stable_psi += pow(beta[l], 2) / lambda_sq[l];
				}
				stable_psi *= (1.0/tau_sq);
				stable_psi = std::max(stable_psi, error_tol);
				gamma_sq[j] = 1.0 / rgig_use(2.0 * eta[j], stable_psi, ((double)grp_size[j] - 1.0) / 2.0);
			} else {
				stable_psi = 0;
				for (int l = 0; l < grp_size_cs[j]; ++l) {
					stable_psi += pow(beta[l], 2) / lambda_sq[l];
				}
				stable_psi *= (1.0 / tau_sq);
				stable_psi = std::max(stable_psi, error_tol);
				gamma_sq[j] = 1.0 / rgig_use(2.0 * eta[j], stable_psi, ((double)grp_size[j] - 1.0) / 2.0);
			}

			//Draw lambda^2
			sum_inv_lambda_sq = 0;
			for (int i = 0; i < grp_size[j]; ++i) {
				if (j != 0) {
					lambda_sq[grp_size_cs[j - 1] + i] = 1.0 / R::rgamma(1.0, 1.0 / (eta[j] + pow(beta[grp_size_cs[j - 1] + i], 2) / (2.0 * tau_sq * gamma_sq[j])));
					sum_inv_lambda_sq += (1.0 / lambda_sq[grp_size_cs[j - 1] + i]);
				} else {
					lambda_sq[i] = 1.0 / R::rgamma(1.0, 1.0 / (eta[j] + pow(beta[i], 2) / (2.0 * tau_sq * gamma_sq[j])));
					sum_inv_lambda_sq += (1.0 / lambda_sq[i]);
				}
			}

			//Draw eta
			eta[j] = R::rgamma(a2 + ((double)grp_size[j] + 1.0) / 2.0, 1.0 / (b2 + gamma_sq[j] + sum_inv_lambda_sq));
		}

		//Draw nu
		nu = 1.0 / R::rgamma(1.0, 1.0 / ((1.0 / tau_sq) + (1.0 / sigma_sq)));

		//Draw psi^2
		psi_sq = 1.0 / R::rgamma(a1 + (double)K / 2.0, 1.0 / (b1 + (alpha.t()*alpha).eval()(0, 0) / 2.0));

		++cnt;
		if (cnt % 500 == 0 && verbose) {
			std::cout << cnt << " Burn-in Draws" << std::endl;
		}
	}

	if (verbose) {
		std::cout << "Burn-in Iterations Complete" << std::endl;
	}

	cnt = 0;
	int total_saved = 0;
	while (total_saved < n_samples) {

		//Draw mu
		mu = R::rnorm(sum(Y - C * alpha - X * beta) / (double)n, sqrt(sigma_sq / (double)n));

		//Draw alpha
		alpha_tmp = inv((1.0 / sigma_sq)*CtC + (1.0 / psi_sq)*diagK);
		alpha = arma::mvnrnd((1.0 / sigma_sq) * alpha_tmp * tC * (Y - mu * ones_vec - X * beta), alpha_tmp);

		//Draw beta
		for (int g = 0; g < M; ++g) {
			local_param_expand_inv[g] = 1.0 / (gamma_sq[grp_idx[g] - 1] * lambda_sq[g]);
		}
		local_param_inv.diag() = local_param_expand_inv;
		beta_tmp = inv((1.0 / sigma_sq) * XtX + (1.0 / tau_sq) * local_param_inv);
		beta = arma::mvnrnd((1.0 / sigma_sq) * beta_tmp * tX * (Y - mu * ones_vec - C * alpha), beta_tmp);

		//Draw tau^2
		tau_sq = 1.0 / R::rgamma(((double)M + 1.0) / 2.0, 1.0 / ((beta.t() * local_param_inv * beta) / 2.0 + 1.0 / nu).eval()(0, 0));

		//Draw sigma^2
		sigma_sq = 1.0 / R::rgamma(((double)n + 1.0) / 2.0, 1.0 / (((Y - mu * ones_vec - C * alpha - X * beta).t()*(Y - mu * ones_vec - C * alpha - X * beta)) / 2.0 + 1.0 / nu).eval()(0, 0));

		for (int j = 0; j < J; ++j) {
			//Draw gamma^2
			if (j != 0) {
				stable_psi = 0;
				for (int l = grp_size_cs[j - 1]; l < grp_size_cs[j]; ++l) {
					stable_psi += pow(beta[l], 2) / lambda_sq[l];
				}
				stable_psi *= (1.0 / tau_sq);
				stable_psi = std::max(stable_psi, error_tol);
				gamma_sq[j] = 1.0 / rgig_use(2.0 * eta[j], stable_psi, ((double)grp_size[j] - 1.0) / 2.0);
			}
			else {
				stable_psi = 0;
				for (int l = 0; l < grp_size_cs[j]; ++l) {
					stable_psi += pow(beta[l], 2) / lambda_sq[l];
				}
				stable_psi *= (1.0 / tau_sq);
				stable_psi = std::max(stable_psi, error_tol);
				gamma_sq[j] = 1.0 / rgig_use(2.0 * eta[j], stable_psi, ((double)grp_size[j] - 1.0) / 2.0);
			}

			//Draw lambda^2
			sum_inv_lambda_sq = 0;
			for (int i = 0; i < grp_size[j]; ++i) {
				if (j != 0) {
					lambda_sq[grp_size_cs[j - 1] + i] = 1.0 / R::rgamma(1.0, 1.0 / (eta[j] + pow(beta[grp_size_cs[j - 1] + i], 2) / (2.0 * tau_sq * gamma_sq[j])));
					sum_inv_lambda_sq += (1.0 / lambda_sq[grp_size_cs[j - 1] + i]);
				}
				else {
					lambda_sq[i] = 1.0 / R::rgamma(1.0, 1.0 / (eta[j] + pow(beta[i], 2) / (2.0 * tau_sq * gamma_sq[j])));
					sum_inv_lambda_sq += (1.0 / lambda_sq[i]);
				}
			}

			//Draw eta
			eta[j] = R::rgamma(a2 + ((double)grp_size[j] + 1.0) / 2.0, 1.0 / (b2 + gamma_sq[j] + sum_inv_lambda_sq));
		}

		//Draw nu
		nu = 1.0 / R::rgamma(1.0, 1.0 / ((1.0 / tau_sq) + (1.0 / sigma_sq)));

		//Draw psi^2
		psi_sq = 1.0 / R::rgamma(a1 + (double)K / 2.0, 1.0 / (b1 + (alpha.t()*alpha).eval()(0, 0) / 2.0));

		//Save output
		if (cnt % n_thin == 0) {
			mu_store[total_saved] = mu;
			alpha_store.row(total_saved) = alpha.t();
			beta_store.row(total_saved) = beta.t();
			lambda_store.row(total_saved) = lambda_sq.t();
			gamma_store.row(total_saved) = gamma_sq.t();
			tau_store[total_saved] = tau_sq;
			sigma_store[total_saved] = sigma_sq;
			eta_store.row(total_saved) = eta.t();
			psi_store[total_saved] = psi_sq;
			nu_store[total_saved] = nu;
			++total_saved;
			if (total_saved % 500 == 0 && verbose) {
				std::cout << total_saved << " Samples Drawn" << std::endl;
			}
		}

		++cnt;
	}

	return(List::create(Named("mus") = mu_store, Named("alphas") = alpha_store, Named("betas") = beta_store, Named("lambda_sqs") = lambda_store,
		Named("gamma_sqs") = gamma_store, Named("tau_sqs") = tau_store, Named("sigma_sqs") = sigma_store, Named("etas") = eta_store, Named("psi_sqs") = psi_store,
		Named("hyperparam_a1") = a1, Named("hyperparam_b1") = b1, Named("hyperparam_a2") = a2, Named("hyperparam_b2") = b2, Named("X") = X, Named("C") = C, Named("Y") = Y,
		Named("grp_idx") = grp_idx, Named("n_burn_in") = n_burn_in, Named("n_samples") = n_samples, Named("n_thin") = n_thin));
}