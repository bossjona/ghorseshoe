// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// rgig_use
double rgig_use(double chi, double psi, double lambda);
RcppExport SEXP _ghorseshoe_rgig_use(SEXP chiSEXP, SEXP psiSEXP, SEXP lambdaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type chi(chiSEXP);
    Rcpp::traits::input_parameter< double >::type psi(psiSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    rcpp_result_gen = Rcpp::wrap(rgig_use(chi, psi, lambda));
    return rcpp_result_gen;
END_RCPP
}
// grouped_horseshoe_gibbs_sampler
List grouped_horseshoe_gibbs_sampler(arma::mat& X, arma::mat& C, arma::colvec& Y, arma::rowvec& grp_idx, arma::rowvec& grp_size, arma::rowvec& grp_size_cs, arma::colvec& alpha_inits, arma::colvec& beta_inits, arma::colvec& lambda_sq_inits, arma::colvec& gamma_sq_inits, arma::colvec& eta_inits, double mu_init, double tau_sq_init, double sigma_sq_init, double psi_sq_init, double nu_init, int n_burn_in, int n_samples, int n_thin, double error_tol, double a1, double b1, double a2, double b2, bool verbose);
RcppExport SEXP _ghorseshoe_grouped_horseshoe_gibbs_sampler(SEXP XSEXP, SEXP CSEXP, SEXP YSEXP, SEXP grp_idxSEXP, SEXP grp_sizeSEXP, SEXP grp_size_csSEXP, SEXP alpha_initsSEXP, SEXP beta_initsSEXP, SEXP lambda_sq_initsSEXP, SEXP gamma_sq_initsSEXP, SEXP eta_initsSEXP, SEXP mu_initSEXP, SEXP tau_sq_initSEXP, SEXP sigma_sq_initSEXP, SEXP psi_sq_initSEXP, SEXP nu_initSEXP, SEXP n_burn_inSEXP, SEXP n_samplesSEXP, SEXP n_thinSEXP, SEXP error_tolSEXP, SEXP a1SEXP, SEXP b1SEXP, SEXP a2SEXP, SEXP b2SEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type C(CSEXP);
    Rcpp::traits::input_parameter< arma::colvec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::rowvec& >::type grp_idx(grp_idxSEXP);
    Rcpp::traits::input_parameter< arma::rowvec& >::type grp_size(grp_sizeSEXP);
    Rcpp::traits::input_parameter< arma::rowvec& >::type grp_size_cs(grp_size_csSEXP);
    Rcpp::traits::input_parameter< arma::colvec& >::type alpha_inits(alpha_initsSEXP);
    Rcpp::traits::input_parameter< arma::colvec& >::type beta_inits(beta_initsSEXP);
    Rcpp::traits::input_parameter< arma::colvec& >::type lambda_sq_inits(lambda_sq_initsSEXP);
    Rcpp::traits::input_parameter< arma::colvec& >::type gamma_sq_inits(gamma_sq_initsSEXP);
    Rcpp::traits::input_parameter< arma::colvec& >::type eta_inits(eta_initsSEXP);
    Rcpp::traits::input_parameter< double >::type mu_init(mu_initSEXP);
    Rcpp::traits::input_parameter< double >::type tau_sq_init(tau_sq_initSEXP);
    Rcpp::traits::input_parameter< double >::type sigma_sq_init(sigma_sq_initSEXP);
    Rcpp::traits::input_parameter< double >::type psi_sq_init(psi_sq_initSEXP);
    Rcpp::traits::input_parameter< double >::type nu_init(nu_initSEXP);
    Rcpp::traits::input_parameter< int >::type n_burn_in(n_burn_inSEXP);
    Rcpp::traits::input_parameter< int >::type n_samples(n_samplesSEXP);
    Rcpp::traits::input_parameter< int >::type n_thin(n_thinSEXP);
    Rcpp::traits::input_parameter< double >::type error_tol(error_tolSEXP);
    Rcpp::traits::input_parameter< double >::type a1(a1SEXP);
    Rcpp::traits::input_parameter< double >::type b1(b1SEXP);
    Rcpp::traits::input_parameter< double >::type a2(a2SEXP);
    Rcpp::traits::input_parameter< double >::type b2(b2SEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(grouped_horseshoe_gibbs_sampler(X, C, Y, grp_idx, grp_size, grp_size_cs, alpha_inits, beta_inits, lambda_sq_inits, gamma_sq_inits, eta_inits, mu_init, tau_sq_init, sigma_sq_init, psi_sq_init, nu_init, n_burn_in, n_samples, n_thin, error_tol, a1, b1, a2, b2, verbose));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_ghorseshoe_rgig_use", (DL_FUNC) &_ghorseshoe_rgig_use, 3},
    {"_ghorseshoe_grouped_horseshoe_gibbs_sampler", (DL_FUNC) &_ghorseshoe_grouped_horseshoe_gibbs_sampler, 25},
    {NULL, NULL, 0}
};

RcppExport void R_init_ghorseshoe(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
