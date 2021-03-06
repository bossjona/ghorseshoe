% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{rgig_use}
\alias{rgig_use}
\title{Function that calls the rgig function from the GeneralizedHyperbolic package in R.}
\usage{
rgig_use(chi, psi, lambda)
}
\arguments{
\item{chi}{A positive double.}

\item{psi}{A positive double.}

\item{lambda}{A non-negative double.}
}
\value{
A random draw from the generalized inverse gaussian distribution with parameters chi, psi, and lambda (double).
}
\description{
Randomly generates one draw from a generalized inverse gaussian distribution.
}
