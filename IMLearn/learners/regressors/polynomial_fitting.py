from __future__ import annotations
from typing import NoReturn
import numpy as np

from . import LinearRegression
###from sklearn.linear_model import LinearRegression###
from ...base import BaseEstimator
###from sklearn.base import BaseEstimator###


class PolynomialFitting(BaseEstimator):
    """
    Polynomial Fitting using Least Squares estimation
    """
    def __init__(self, k: int) -> PolynomialFitting:
        """
        Instantiate a polynomial fitting estimator

        Parameters
        ----------
        k : int
            Degree of polynomial to fit
        """
        super().__init__()
        self.k_ = k
        # We don't ask to use include_intercept. we have it already in vandermonde x^0
        self.linear_regression_ = LinearRegression(include_intercept=False)

    LinearRegression()
    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to polynomial transformed samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        #DOESNT WORK: X_van = _PolynomialFitting__transform(X) # Vandermonde matrix of X in order of self.k_
        X_van = np.vander(X, self.k_+1)
        self.linear_regression_.fit(X_van, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        #DOESNT WORK: X_van = _PolynomialFitting__transform(X) # Vandermonde matrix of X in order of self.k_
        X_van = np.vander(X, self.k_+1)
        y_hat = self.linear_regression_.predict(X_van)
        return y_hat

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """

        #DOESNT WORK: X_van = _PolynomialFitting__transform(X) # Vandermonde matrix of X in order of self.k_
        X_van = np.vander(X, self.k_+1)
        mse = self.linear_regression_.loss(X_van, y)
        return mse

    def __transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform given input according to the univariate polynomial transformation

        Parameters
        ----------
        X: ndarray of shape (n_samples,)

        Returns
        -------
        transformed: ndarray of shape (n_samples, k+1)
            Vandermonde matrix of given samples up to degree k
        """
        if 0: # if not using np.vander
            X = X.reshape(-1,1)
            ones = np.ones(np.shape(X))
            if self.k_==0: return  ones
            VAN = np.append( X, ones, axis=1 )
            for _ in range(self.k_-1):
                new_col = VAN[:,-1].reshape(-1,1) * X
                VAN = np.append(new_col, VAN, axis=1)
        else:
            VAN = np.vander(X,self.k_+1)
        return VAN
