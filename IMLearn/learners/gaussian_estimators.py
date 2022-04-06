from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet

# Implemented in Ex1

class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """

        self.mu_ = 1/len(X) * np.sum(X)
        if self.biased_ :
            self.var_ = 1/len(X) * np.sum( (X - self.mu_) **2 )
        else:
            self.var_ = 1/(len(X)-1) * np.sum( (X - self.mu_) **2 )

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        stddiv = np.sqrt(self.var_)
        pdf = 1 / ( np.sqrt(2 * np.pi) * stddiv ) * \
                    np.exp( - ( X - self.mu_) ** 2 / (2 * stddiv ** 2) )

        return pdf

        # The full graph of a PDF:
        #sigma = np.sqrt(self.var_)
        #bins = round(np.sqrt(len(X)))
        #range_tuple = (self.mu_ - 5*sigma , self.mu_ + 5*sigma )
        #hist, bins_edges = np.histogram(X, bins=bins, range=range_tuple )
        #pdf = hist / len(X) * bins / (range_tuple[1] - range_tuple[0])
        #print("integral pdf samples "+str(sum(pdf) * (bins_edges[1]-bins_edges[0]) ) )

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        Xmm = X - mu
        m = len(X)
        loglike = m * ( - 0.5*np.log(2*np.pi) - np.log(sigma) ) - np.sum( Xmm * Xmm ) / 2*sigma**2

        return loglike

class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """

        self.mu_ =  1/np.size(X, axis=0) * np.sum(X, axis=0)

        # make a 2D array of mu (of each feature) per sample
        mu2d = [self.mu_] * np.size(X, axis=0)
        # X minus mu (expected values) to have only offsets from expected estimated value
        Xmm = X - mu2d
        # make 3D array having the new axis (num 2) in size of axis 1 (also size of mu)
        X3mm = np.repeat( Xmm[:,:,np.newaxis], np.size(X,1), axis=2 )
        X3mmT = np.transpose(X3mm, axes=(0,2,1))

        # The "in-place" X3mm * X3mmT will result in M[i,j,smp] = f_i*f_j of the sample
        # so averging along the 0 axis will give the Covariance of x_i with x_j
        self.cov_ = 1/(np.size(X,0)-1) * np.sum( X3mm * X3mmT  ,axis=0)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        # make a 2D array of mu (of each feature) per sample
        mu2d = [self.mu_] * np.size(X, axis=0)
        # X minus mu (expected values) to have only offsets from expected estimated value
        Xmm = X - mu2d

        samples_pdf = []
        for smp in Xmm:
            pdf = 1 / ( np.sqrt(2 * np.pi) * np.linalg.det(self.cov_) ) * \
                        np.exp( -0.5 * smp @ self.cov_ @ smp.T )
            samples_pdf.append(pdf)

        return np.array(samples_pdf)

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """

        # make a 2D array of mu (of each feature) per sample
        mu2d = [mu] * np.size(X, axis=0)
        # X minus mu (expected values) to have only offsets from expected estimated value
        Xmm = X - mu2d

        _, logdet_cov = np.linalg.slogdet(cov)
        loglike_coeff = - 0.5*np.log(2*np.pi) - logdet_cov
        accumulate = 0.0
        if False : # The clear but heavy computation method
            for smp in Xmm:
                loglike = loglike_coeff - 0.5 * smp @ np.linalg.inv(cov) @ smp.T
                accumulate = accumulate + loglike
        else: # cumputaion effective calculation
            # per sample feature set we have the multiple by the cov-matrix and then
            # each vector (alone) is multipled by the features vector again.
            accumulate =  len(X) * loglike_coeff - 0.5 * np.sum( (Xmm @ cov) * Xmm ) # like diag(Xmm @ cov @ Xmm)
            # Note the sum is both to complete the last inner-product (on axis 1)
            # and also summing along all samples (on axis 0)

        return accumulate
