\name{ghorseshoe-package}
\alias{ghorseshoe-package}
\alias{ghorseshoe}
\docType{package}
\title{
\packageTitle{ghorseshoe}
}
\description{
\packageDescription{ghorseshoe}
}
\details{

The DESCRIPTION file:
\packageDESCRIPTION{ghorseshoe}
\packageIndices{ghorseshoe}
This package implements a Gibbs sampler for grouped horseshoe regression. There are three functions that are included in the package. The primary function is the grouped_horseshoe function which only requires four inputs: (1) the outcome vector (Y), (2) the part of the design matrix corresponding to the adjustment covariates (C), (3) the part of the design matrix corresponding to the variables we want to implement grouped horseshoe shrinkage on (X), and (4) the grp_idx argument which lists which covariate group each column of X belongs. For the function to work properly, make sure that group indicies in grp_idx take the values 1,2,...,J (where J is the total number of covariate groups in X) and that the columns of X are ordered by grp_idx (that is, all the columns in X corresponding to group 1 come first, followed by all the columns in X corresponding to group 2, and so on). The grouped_horseshoe_gibbs_sampler function is called by the grouped_horseshoe function and runs the grouped horsehoe regression Gibbs sampler. The rgig_use function calls the rgig function from the GeneralizedHyperbolic package in R which is used in the grouped_horseshoe_gibbs_sampler function. Therefore, the only function that the user needs to use is the grouped_horseshoe function.
}
\author{
\packageAuthor{ghorseshoe}

Maintainer: \packageMaintainer{ghorseshoe}
}
\keyword{ package }
