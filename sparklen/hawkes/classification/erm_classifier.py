# Author: Romain E. Lacoste
# License: BSD-3-Clause

import time

import numpy as np
from tqdm import tqdm

from sparklen.hawkes.model import ModelHawkesExpClassification
from sparklen.hawkes.model.build.hawkes_model import (
    ModelHawkesExpLogLikelihoodSingle as CppModelHawkesExpLogLikelihoodSingle,
)
from sparklen.plot import plot_confusion_matrix


class ERMCLassifier():
    """
    ERM classifier class for Hawkes process with exponential kernel.

    This class performs classification for a Hawkes process based on observed labeled data.

    The observed data consist of labeled repeated realizations of a Hawkes process.

    .. math::
        \\left\\{\\left(\\left\\{t_{j,\\ell}^{(i)}\\right\\}_{\\ell \\in [N_j(T)]},
        Y^{(i)}\\right), \ j \\in [d], \ i \\in [n] \\right\}

    where

    * :math:`K \\geq 2` is the number of classes
    * :math:`n \\geq 1` is the number of repetitions
    * :math:`d \\geq 1` is the number of components
    * :math:`T > 0` is the time horizon
    * :math:`t_{j,\\ell}^{(i)}` is the :math:`\\ell`-th event time of component :math:`j` in the :math:`i`-th repetition
    * :math:`Y^{(i)} \\in [K]` is the label of the :math:`i`-th feature

    Parameters
    ----------
    decay : float
        The decay hyperparameter of the exponential kernel of the process.
        This scalar dictates how quick the influences vanish over time.

    gamma0 : float, default=0.1
        Input of `Free Adagrad` optimizer.
        Initial lower-bound guess for the distance between the starting point
        and the optimum

    max_iter : int, default=100
        The maximum number of iterations allowed during the optimization process.

    tol : float, default=1e-5
        The tolerance for the precision achieved during the optimization process.
        The optimization will stop when the convergence criterion falls below this value.
        If the tolerance is not reached, the optimizer will perform a maximum of `max_iter` iterations.

    verbose_bar : bool, default=True
        Determines whether a progress bar is displayed during the optimization process,
        along with information such as the loss value and convergence criterion.
        If `verbose_bar=False`, no information is displayed.
        If set to `True`, details will be displayed every `print_every` iterations.

    verbose : bool, default=True
        Controls whether recorded information during the optimization phase is printed at the end.
        If `verbose=False`, no information is printed.
        If set to `True`, details will be displayed every `print_every` iterations.

    print_every : int, default=10
        Specifies the frequency at which history information is printed.
        Information will be printed when the iteration number is a multiple of `print_every`.
    """

    def __init__(self, decay, gamma0=0.1, max_iter=100, tol=1e-5,
                 verbose_bar=True, verbose=True, print_every=10):

        self._decay = decay
        self._gamma0 = gamma0
        self._max_iter = max_iter
        self._tol = tol
        self._verbose_bar = verbose_bar
        self._verbose = verbose
        self._print_every = print_every

        self._estimated_params = None
        self._is_fitted = False

    def _initialize_values(self, X, y, end_time):
        K = np.max(y)+1
        M = len(X[0])

        self._weights = np.empty(K)
        for k in range(K):
            self._weights[k] = np.mean(y == k)

        x0 = np.ones((K, M, M + 1)) * 0.1

        model_train = ModelHawkesExpClassification(self._decay, self._weights)
        model_train.set_data(X, y, end_time)

        S = 0.0
        Gamma = 0.0
        k = 1
        gamma = self._gamma0
        x = x0

        loss_x = model_train.loss(x)
        grad_x = model_train.grad(x)

        return model_train, S, Gamma, k, gamma, x0, x, loss_x, grad_x

    @staticmethod
    def _proj_operator(bold_theta):
        K, M = bold_theta.shape[:2]

        for k in range(K):
            for i in range(0, M):
                for j in range(0, M):
                    bold_theta[k][i][j+1] = np.minimum(1., np.maximum(bold_theta[k][i][j+1], 0.))

        return bold_theta

    @staticmethod
    def _step(x, eta, g):
        return ERMCLassifier._proj_operator(x-eta*g)


    def fit(self, X, y, end_time=None):
        """
        Build an ERM classifier from the training set `(X, y)`.

        Parameters
        ----------
        X : list of list of ndarray
            The training input samples. The outer list has length `n`,
            representing repetitions. Each inner list has length `d`, where
            each element is a one-dimensional ndarray containing the
            event times of a specific component.

        y : ndarray of shape (n,)
            The class labels for each training sample. Each entry is an
            integer representing the class membership.

        end_time : float, default=None
            The end time of the observation period. The time horizon defines
            the interval `[0, T]` over which the Hawkes process is observed.

            - If `end_time` is provided, it is used as the upper bound of the observation window.
            - If `end_time=None`, it is automatically set to the largest observed event time
              across all components and repetitions.

        Returns
        -------
        self : object
            The instance of the fitted model.
        """
        start_time = time.time()  # Start the timer

        # Initialize values
        model_train, S, Gamma, k, gamma, x0, x, loss_x, grad_x = self._initialize_values(X, y, end_time)

        pbar = None
        if self._verbose_bar:
            # Setup the progress bar
            pbar = tqdm(total=self._max_iter, desc="Training", unit="it")

        try:
            for iteration in range(self._max_iter):

                norm_grad_x = np.linalg.norm(grad_x)
                S += norm_grad_x ** 2
                h = np.sqrt((S + 1.0) * (1.0 + np.log(1.0 + S)))
                while True :
                    x_new = ERMCLassifier._step(x, gamma/h, grad_x)
                    B = (2.0 / np.sqrt(k)) * gamma + np.sqrt(Gamma + (gamma * norm_grad_x / h) ** 2)
                    if np.linalg.norm(x_new - x0) > B:
                        k += 1
                        gamma *= 2
                    else :
                        Gamma += (gamma * (norm_grad_x / h)) ** 2
                        break
                loss_x_new = model_train.loss(x_new)
                grad_x_new = model_train.grad(x_new)

                # Update relative distance
                rel_loss = abs(loss_x_new - loss_x) / abs(loss_x)

                # Update progress bar and print detailed information based on print_every
                if self._verbose_bar and iteration % self._print_every == 0:
                    pbar.update(self._print_every)
                    if self._verbose:
                        pbar.set_postfix({"Loss": loss_x_new, "gamma": gamma})

                # Check for convergence
                converged = rel_loss < self._tol
                if converged:
                    break

                # Update x, loss_x, and grad_x for the next iteration
                x, loss_x, grad_x = x_new, loss_x_new, grad_x_new

        except Exception as e:
            if pbar:
                pbar.write(f"\nTraining interrupted: {e}")

        finally:
            end_time = time.time()  # End the timer
            self._elapsed_time = end_time - start_time

            if pbar:
                # Print the status message
                if converged:
                    pbar.write(f"\nTraining completed. Convergence achieved after {iteration + 1} iterations.")
                else:
                    pbar.write(f"\nTraining terminated. Max iterations {self._max_iter} reached.")
                pbar.write(f"\nTime elapsed: {self._elapsed_time:.2f} seconds.")

                pbar.close()

        self._estimated_params = x

        self._is_fitted = True

    def predict(self, X, end_time=None):
        """
        Predict class labels for `X`.

        The predicted class of an input sample is determined by the ERM classifier,
        selecting the class with the highest predictive probability.

        This method requires that the `fit` method has been called
        beforehand to build the ERM classifier.

        Parameters
        ----------
        X : list of list of ndarray
            The input samples. The outer list has length `n`, representing
            repetitions. Each inner list has length `d`, where each element
            is a one-dimensional `ndarray` containing the event times of a
            specific component.

        end_time : float, default=None
            The end time of the observation period. The time horizon defines
            the interval `[0, T]` over which the Hawkes process is observed.

            - If `end_time` is provided, it is used as the upper bound of the observation window.
            - If `end_time=None`, it is automatically set to the largest observed event time
              across all components and repetitions.

        Returns
        -------
        y_pred : ndarray of shape (n,)
            The predicted class labels for each input sample.
        """
        probabilities = self.predict_proba(X, end_time)
        return np.argmax(probabilities, axis=1)

    def predict_proba(self, X, end_time=None):
        """
        Predict class probabilities for X.

        This method requires that the `fit` method has been called
        beforehand to build the ERM classifier.

        Parameters
        ----------
        X : list of list of ndarray
            The input samples. The outer list has length `n`, representing
            repetitions. Each inner list has length `d`, where each element
            is a one-dimensional `ndarray` containing the event times of a
            specific component.

        end_time : float, default=None
            The end time of the observation period. The time horizon defines
            the interval `[0, T]` over which the Hawkes process is observed.

            - If `end_time` is provided, it is used as the upper bound of the observation window.
            - If `end_time=None`, it is automatically set to the largest observed event time
              across all components and repetitions.

        Returns
        -------
        probabilities : ndarray of shape (n,K)
            The class probabilities of the input samples.
        """
        if not self._is_fitted:
                raise ValueError("Training has not been completed. You must call fit() before getting the predict class probabilities.")

        if end_time is None:
            end_time = max([max((np.max(events, initial=0.0) for events in inner), default=0.0) for inner in X])

        n = len(X)
        K, M = self._estimated_params.shape[:2]

        probabilities = np.empty((n,K), dtype=np.float64)

        for rep in range(n):
            for k in range(K):
                model = CppModelHawkesExpLogLikelihoodSingle(M)
                F_k = model.compute_loss(X[rep], end_time, self._decay, self._estimated_params[k], False)
                if (F_k >= np.log(np.finfo(np.float64).max)):
                    F_k = np.log(np.finfo(np.float64).max)
                probabilities[rep][k] = self._weights[k] * np.exp(F_k)
            probabilities[rep, :] /= np.sum(probabilities[rep, :])
        return probabilities

    def score(self, X, y, end_time=None):
        """
        Return the mean accuracy on the given test data and labels.

        This method requires that the `fit` method has been called
        beforehand to build the ERM classifier.

        Parameters
        ----------
        X : list of list of ndarray
            The test input samples. The outer list has length `n`, representing
            repetitions. Each inner list has length `d`, where each element
            is a one-dimensional `ndarray` containing the event times of a
            specific component.

        y : ndarray of shape (n,)
            The true class labels for each test sample.

        end_time : float, default=None
            The end time of the observation period. The time horizon defines
            the interval `[0, T]` over which the Hawkes process is observed.

            - If `end_time` is provided, it is used as the upper bound of the observation window.
            - If `end_time=None`, it is automatically set to the largest observed event time
              across all components and repetitions.

        Returns
        -------
        accuracy : float
            The mean accuracy of the classifier on the test set.
        """
        y_pred = self.predict(X, end_time)

        return np.mean(y_pred == y)

    def plot_score_cm(self, X, y, end_time=None, save_path=None, save_format='png', dpi=300, use_latex=False):
        """
        Plot the confusion matrix to visualize the classification performance
        of the ERM classifier.

        The confusion matrix compares true labels with predicted labels,
        showing counts of correct and incorrect predictions.

        This method requires that the `fit` method has been called
        beforehand to build the ERM classifier.

        This method calls the plot function :function:`~sparklen.plot.plot_hawkes.plot_confusion_matrix`


        Parameters
        ----------
        X : list of list of ndarray
            The test input samples. The outer list has length `n`, representing
            repetitions. Each inner list has length `d`, where each element
            is a one-dimensional `ndarray` containing the event times of a
            specific component.

        y : ndarray of shape (n,)
            The true class labels for each test sample.

        end_time : float, default=None
            The end time of the observation period. The time horizon defines
            the interval `[0, T]` over which the Hawkes process is observed.

            - If `end_time` is provided, it is used as the upper bound of the observation window.
            - If `end_time=None`, it is automatically set to the largest observed event time
              across all components and repetitions.

        save_path : str, optional, default=None
            The path where the plot will be saved. If not provided, the plot will not be saved.

        save_format : str, optional, default='png'
            The format in which to save the plot (e.g., 'png', 'pdf', 'pgf').

        dpi : int, optional, default=300
            The resolution of the saved plot. Higher values result in higher quality.

        use_latex : bool, optional, default=False
            Whether to use LaTeX for rendering text in the plot.
            If `True`, text will be rendered using LaTeX formatting.
        """
        y_pred = self.predict(X, end_time)

        plot_confusion_matrix(y, y_pred, save_path, save_format, dpi, use_latex)

    def __repr__(self):
        class_name = self.__class__.__name__
        return (f"{class_name}(decay={self._decay}, gamma0={self._gamma0}, max_iter={self._max_iter}, tol={self._tol}"
                f"verbose_bar={self._verbose_bar}, verbose={self._verbose}, print_every={self._print_every})")
