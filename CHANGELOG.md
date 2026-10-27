# Changelog

All notable changes to Sparklen are documented in this file.

## [Unreleased]

### Added
- `sparklen.__version__`.
- Automated tests on Linux, macOS and Windows, and an automated release of
  ready-to-install wheels for all platforms.

### Changed
- Python 3.10 or later is required; Python 3.13 and 3.14 are supported.
- Installing from source no longer requires SWIG on the system
  (#1, thanks to @tomMoral).

### Removed
- `pandas` from the dependencies (it was never used).

### Fixed
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