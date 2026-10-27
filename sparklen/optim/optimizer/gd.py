# Author: Romain E. Lacoste
# License: BSD-3-Clause

import time

from tabulate import tabulate
from tqdm import tqdm

from sparklen.optim.optimizer.base.optimizer import Optimizer


class GD(Optimizer):
    """
    Optimizer class for Gradient Descent.

    This class implements the Gradient Descent optimization method, which is
    often regarded as the workhorse of first-order optimization, known for
    its simplicity and effectiveness across a broad range of application.

    Parameters
    ----------
    lr_scheduler : str, {'lipschitz', 'backtracking'}, default='backtracking'
        Specifies the learning rate scheduler. The available options are:

        - 'lipschitz' : Lipschitz-based step size, usable only if the loss is gradient Lipschitz.
        - 'backtracking' : Backtracking line-search-based step size.
        - 'fast-backtracking' : Two-Way Backtracking line-search-based step size.

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

    print_every : int, default=5
        Specifies the frequency at which history information is printed.
        Information will be printed when the iteration number is a multiple of `print_every`.

    record_every : int, default=5
        Specifies the frequency at which history information is recorded.
        Information will be recorded when the iteration number is a multiple of `record_every`.

    Attributes
    ----------
    minimizer : ndarray
        Minimizer found by the optimizer. This is a read-only property.

    elapsed_time : float
        Time taken for the `optimize()` call, in seconds. This is a read-only property.

    history : dict-like
        A dictionary-like object storing the optimizer's history across iterations.
        This is a read-only property.
    """

    def __init__(self, lr_scheduler, max_iter, tol, verbose_bar=True, verbose=True, print_every=10, record_every=1):
        # Call the initializer of the base class Optimizer
        super().__init__(lr_scheduler, max_iter, tol, verbose_bar, verbose, print_every, record_every)
        #self._cpp_gd = CppGD()

    def _initialize_values(self, x0):
        """Initialize loss_x and grad_x based on the initial parameters x."""
        x = x0
        loss_x = self._model.loss(x)
        grad_x = self._model.grad(x)
        return x, loss_x, grad_x

    def _step(self, x, loss_x, grad_x):
        """
        Perform a single optimization step based on the current
        parameters and gradients.
        """
        step_size, x_new, loss_x_new, grad_x_new = self._lr_scheduler.step(x, loss_x, grad_x)

        return step_size, x_new, loss_x_new, grad_x_new

    def optimize(self, x0):
        """
        Run the optimization process.


        Parameters
        ----------
        x0 : ndarray
            Initial guess.

        Returns
        -------
        self : object
            The instance of the optimized object.
        """

        start_time = time.time()  # Start the timer

        # Call the base class logic
        super().optimize(x0)

        # Initialize the loss and gradient for the first iteration
        x, loss_x, grad_x = self._initialize_values(x0)

        pbar = None
        if self._verbose_bar:
            # Setup the progress bar
            pbar = tqdm(total=self._max_iter, desc="Optimizing", unit="it")

        try:
            for iteration in range(self._max_iter):
                step_size, x_new, loss_x_new, grad_x_new = self._step(x, loss_x, grad_x)

                # Update relative distance
                rel_loss = abs(loss_x_new - loss_x) / abs(loss_x)

                # Update progress bar and print detailed information based on print_every
                if self._verbose_bar and iteration % self._print_every == 0:
                    pbar.update(self._print_every)
                    if self._verbose:
                        pbar.set_postfix({"loss": loss_x_new, "lr": step_size})

                # Record history based on record_every
                if iteration % self._record_every == 0:
                    self.record_history(x_new, loss_x_new, grad_x_new, step_size, rel_loss, iteration)

                # Check for convergence
                converged = rel_loss < self._tol
                if converged:
                    break

                # Update x, loss_x, and grad_x for the next iteration
                x, loss_x, grad_x = x_new, loss_x_new, grad_x_new

        except Exception as e:
            if pbar:
                pbar.write(f"\nOptimization interrupted: {e}")

        finally:
            end_time = time.time()  # End the timer
            self._elapsed_time = end_time - start_time

            if pbar:
                # Print the status message
                if converged:
                    pbar.write(f"\nOptimization completed. Convergence achieved after {iteration + 1} iterations.")
                else:
                    pbar.write(f"\nOptimization terminated. Max iterations {self._max_iter} reached.")
                pbar.write(f"\nTime elapsed: {self._elapsed_time:.2f} seconds.")

                pbar.close()

        self._is_optimized = True

        # Print full history based on verbose
        if self._verbose:
            self.print_history()

        self._minimizer = x

    def print_info(self):
        """ Display information about the instantiated model object. """
        table = [["Optimizer", "Gradient Descent"],
                 ["Learning rate strategy", self._lr],
                 ["Regularization", self._prox],
                 ["Penalization constant", self._kappa],
                 ["Maximum iteration", self._max_iter],
                 ["Tolerance", self._tol],
                 ["Verbose", self._verbose]]
        print(tabulate(table, headers="firstrow", tablefmt="grid"))
