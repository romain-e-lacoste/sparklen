# Changelog

All notable changes to Sparklen are documented in this file.

## [Unreleased]

## [1.1.1] - 2026-11-08

First release on PyPI since 1.0.0, with wheels for Linux, macOS and Windows.
Several fixes change numerical results: see Fixed.

### Added
- `sparklen.__version__`.
- Automated tests on Linux, macOS and Windows, and an automated release of
  ready-to-install wheels for all platforms.

### Changed
- Python 3.10 or later is required; Python 3.13 and 3.14 are supported.
- Installing from source no longer requires SWIG on the system
  (#1, thanks to @tomMoral).
- Errors during optimisation, calibration and classifier training are raised
  instead of being printed and ignored. A loss that becomes NaN or infinite
  raises `FloatingPointError`. A failed calibration or fit no longer keeps
  the result of an earlier one.
- Event times are checked when data are set: they must be finite,
  non-negative, sorted and not later than `end_time`. Lists and
  non-contiguous arrays are accepted.
- Simultaneous events raise `ValueError`: within a path, event times must be
  distinct, in one component or across components. A Hawkes process has no
  simultaneous events, the least-squares loss assumed none, and estimates
  from rounded timestamps with ties are biased. Add uniform jitter within
  the rounding step, or use finer timestamps.
- Classification labels must be non-negative integers, and the ERM and ERMLR
  classifiers need at least two training paths.
- `ModelHawkesExpClassification` checks its class weights: one per class,
  finite and non-negative, with a positive sum.
- A calibration skips a penalisation constant whose score is not finite, with
  a `RuntimeWarning`, and raises `RuntimeError` if no score is finite.
- `SimuHawkesExp.simulate` raises `ValueError` when the spectral radius of the
  interaction matrix is at least one, instead of never returning.
- A line search that cannot find a step size emits a `RuntimeWarning`
  instead of printing a message.
- The `apply` method of the proximal operators returns the updated array, and
  raises `ValueError` for an array that is not float64, C-contiguous and
  writeable, since it works in place.

### Removed
- `pandas` from the dependencies (it was never used).

### Fixed
- A term of the classification-loss gradient had the wrong sign, so the ERM
  and ERMLR classifiers now fit different parameters.
- Class probabilities are computed in log space, so they no longer become NaN
  on long observation windows; this changes `predict_proba` and `predict`.
- The ERM and ERMLR classifiers keep each exogenous intensity at least 1/n,
  where n is the number of training paths, as in the ERMLR paper; before, an
  intensity could reach zero or below and make the training loss NaN. The
  bound depends on the time unit: if the baseline rates are below 1/n,
  measure time in a larger unit.
- The least-squares Hessian had the entries of its first row and column
  shifted by one position, so the Lipschitz constant and the step size of
  `lr_scheduler="lipschitz"` change.
- `lr_scheduler="lipschitz"` returned the starting point unchanged with
  NumPy 2.5 or later.
- The Lipschitz step size is recomputed for every new model or data set,
  for example across cross-validation folds.
- `AGD` uses the FISTA momentum it documents, so its iterates change and
  converge faster.
- Optimisers return the newest iterate on convergence, not the previous one.
- Fortran-ordered or non-contiguous arrays were read in the wrong order.
- Changing `decay` after a first computation had no effect on the results.
- `SimuHawkesExp.spectral_radius` returns a real number: it could return a
  complex number with a zero imaginary part.
- `fit`, `optimize`, `calibrate` and `simulate` return `self`, as documented.
- `from sparklen.calibration import *` and `print_info` of `GD` and `AGD`
  raised an error; `CalibrationCV` with fewer paths than folds now says so;
  the weights check of `make_classification` rejected NumPy integers.
- Arrays returned by the C++ core were freed by NumPy with the wrong
  allocator (undefined behaviour). They are still passed to NumPy without a
  copy, and the C++ owner now frees them once NumPy no longer uses them.
- `SyntaxWarning: invalid escape sequence` when importing with Python 3.12
  or later.
- The citation file is now valid and points to the Sparklen paper.

## [1.1.0] - 2025-04-12

Released on GitHub only, not on PyPI.

### Added
- Implementation of a new learning rate scheduler:
	two-way backtracking line search.
- Added in the C++ model classes: pre-calculation now leverage
	the Markov property of the exponential kernel.

### Fixed
- The `LearnerHawkesExp` is now compatible with scikit-learn routines
	for variable selection, enabling an easy-to-use pipeline for
	selecting the decay hyperparameter when it is unknown.
- Providing `end_time` with data is now optional.

## [1.0.0] - 2025-02-24

### Added
- Initial release of Sparklen with core features:

  * A efficient cluster-based simulation method for generating events.

  * A highly versatile and flexible framework for performing inference of
    multivariate Hawkes process.

  * Novel approaches to address the challenge of multiclass
    classification within the supervised learning framework.

[Unreleased]: https://github.com/romain-e-lacoste/sparklen/compare/1.1.1...HEAD
[1.1.1]: https://github.com/romain-e-lacoste/sparklen/compare/1.1.0...1.1.1
[1.1.0]: https://github.com/romain-e-lacoste/sparklen/compare/1.0.0...1.1.0
[1.0.0]: https://github.com/romain-e-lacoste/sparklen/releases/tag/1.0.0
