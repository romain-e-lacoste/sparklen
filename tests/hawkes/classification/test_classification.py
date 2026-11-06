# Author: Romain E. Lacoste
# License: BSD-3-Clause

"""Classification model and the ERM / ERMLR classifiers."""

import numpy as np
import pytest
from hawkes_numpy_reference import finite_difference_grad, log_likelihood, tiny_paths

from sparklen.hawkes.classification import (
    ERMCLassifier,
    ERMLRCLassifier,
    make_classification,
)
from sparklen.hawkes.model import ModelHawkesExpClassification

END_TIME = 5.0
DECAY = 1.5


@pytest.fixture
def bold_theta():
    return np.array([[[0.4, 0.3, 0.1], [0.2, 0.05, 0.25]],
                     [[0.9, 0.0, 0.2], [0.5, 0.4, 0.0]],
                     [[0.3, 0.1, 0.1], [0.6, 0.2, 0.3]]])


@pytest.fixture
def model():
    X = tiny_paths(n_paths=4, n_components=2, end_time=END_TIME)
    y = np.array([0, 2, 1, 2])
    model = ModelHawkesExpClassification(decay=DECAY, weights=np.array([0.2, 0.3, 0.5]))
    model.set_data(X, y, END_TIME)
    return model


def test_loss_matches_the_reference(model, bold_theta):
    """Check that the classification loss matches its definition, computed from
    the NumPy log-likelihood reference."""
    expected = 0.0
    for path, label in zip(model.X, model.y):
        F = np.array([log_likelihood([path], END_TIME, DECAY, theta) for theta in bold_theta])
        pi = model.weights * np.exp(F)
        pi /= pi.sum()
        expected += np.sum((pi - np.eye(3)[label]) ** 2)
    # The loss is 4/n times the sum of the squared distances over the n paths
    assert model.loss(bold_theta) == pytest.approx(4 * expected / len(model.X), rel=1e-9)


@pytest.mark.parametrize("point", ["fixture", "random"])
def test_grad_matches_finite_differences(model, bold_theta, point):
    """Check that the classification gradient matches finite differences,
    at the fixture point and at a random positive point."""
    if point == "random":
        bold_theta = np.random.default_rng(0).uniform(0.05, 0.6, bold_theta.shape)
    expected = finite_difference_grad(model.loss, bold_theta)
    np.testing.assert_allclose(model.grad(bold_theta), expected, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("F", [np.array([-1e5, -1e5 + 3.0, -1e5 - 2.0]),
                               np.array([1e3, 1e3 + 1.0, 1e3 - 4.0])])
def test_posterior_is_computed_in_log_space(F):
    """Check that the posterior stays accurate for very large or very small
    log-likelihoods."""
    p = np.array([0.2, 0.3, 0.5])
    shifted = F - F.max()
    expected = p * np.exp(shifted) / np.sum(p * np.exp(shifted))
    pi = ModelHawkesExpClassification._posterior(F, p)
    np.testing.assert_allclose(pi, expected, rtol=1e-9)


def test_posterior_handles_an_empty_class():
    """Check that a class of weight 0 gets posterior 0 without producing NaN."""
    pi = ModelHawkesExpClassification._posterior(np.array([1.0, 2.0]), np.array([1.0, 0.0]))
    np.testing.assert_array_equal(pi, [1.0, 0.0])


def test_weights_must_match_the_number_of_classes(bold_theta):
    """Check that weights of the wrong length for the number of classes
    raise a ValueError."""
    model = ModelHawkesExpClassification(decay=DECAY, weights=np.array([0.5, 0.5]))
    model.set_data(tiny_paths(n_paths=3), np.array([0, 1, 2]), END_TIME)
    with pytest.raises(ValueError, match="length of weights"):
        model.loss(bold_theta)


@pytest.mark.parametrize("weights", [[0.0, 0.0, 0.0], [0.5, -0.2, 0.7], [0.5, np.nan, 0.5]])
def test_weights_must_be_non_negative_with_a_positive_sum(weights, bold_theta):
    """Check that zero, negative or NaN weights raise a ValueError
    instead of giving NaN probabilities."""
    model = ModelHawkesExpClassification(decay=DECAY, weights=np.array(weights))
    model.set_data(tiny_paths(n_paths=3), np.array([0, 1, 2]), END_TIME)
    with pytest.raises(ValueError, match="non-negative"):
        model.loss(bold_theta)


@pytest.mark.parametrize("cls", [ERMCLassifier, ERMLRCLassifier])
def test_projection_onto_the_parameter_set(cls):
    """Check that the projection keeps mu at least 1/n, without an upper bound,
    and clips the interactions to [0, 1]."""
    n = 50
    bold_theta = np.array([[[-1.0, -0.5, 2.0], [10.0, 0.3, 0.7]]])
    projected = cls._proj_operator(bold_theta.copy(), n)
    np.testing.assert_allclose(projected[0, :, 0], [1 / n, 10.0])
    np.testing.assert_allclose(projected[0, :, 1:], [[0.0, 1.0], [0.3, 0.7]])


def make_sample(n_samples, random_state):
    bold_mu = np.array([[0.5, 0.5], [1.0, 0.2]])
    bold_alpha = np.array([[[0.3, 0.0], [0.0, 0.3]],
                           [[0.0, 0.4], [0.4, 0.0]]])
    return make_classification(bold_mu, bold_alpha, beta=2.0, end_time=20.0,
                               n_samples=n_samples, n_classes=2,
                               random_state=random_state)


@pytest.fixture(scope="module")
def labelled_sample():
    return make_sample(30, random_state=0)


@pytest.fixture(scope="module")
def held_out_sample():
    return make_sample(200, random_state=1)


@pytest.mark.parametrize("cls", [ERMCLassifier, ERMLRCLassifier])
def test_fit_and_predict(cls, labelled_sample, held_out_sample, capsys):
    """Check that fit returns self without printing, that the outputs are valid,
    and that the accuracy on held-out paths exceeds 0.85."""
    X, y = labelled_sample
    classifier = cls(decay=2.0, max_iter=20, verbose_bar=False, verbose=False)
    assert classifier.fit(X, y, end_time=20.0) is classifier
    assert capsys.readouterr().out == ""  # no stray debug print

    proba = classifier.predict_proba(X, end_time=20.0)
    assert proba.shape == (len(X), 2)
    assert np.all(np.isfinite(proba))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    params = classifier._estimated_params
    assert np.all(params[:, :, 0] >= 1 / len(X))
    assert np.all(np.isfinite(params))
    X_new, y_new = held_out_sample
    assert classifier.score(X_new, y_new, end_time=20.0) > 0.85



@pytest.mark.parametrize("cls", [ERMCLassifier, ERMLRCLassifier])
def test_failed_fit_leaves_no_model(cls, labelled_sample):
    """Check that a failed fit does not keep the model of an earlier one."""
    X, y = labelled_sample
    classifier = cls(decay=2.0, max_iter=20, verbose_bar=False, verbose=False)
    classifier.fit(X, y, end_time=20.0)
    with pytest.raises(ValueError, match="non-negative"):
        classifier.fit(X, -y, end_time=20.0)
    with pytest.raises(ValueError, match="call fit"):
        classifier.predict_proba(X, end_time=20.0)

def test_predict_proba_on_long_paths_has_no_nan():
    """Check that predict_proba stays finite on paths of a few thousand events,
    whose log-likelihoods used to underflow to 0/0."""
    bold_mu = np.array([[3.0, 3.0], [3.5, 2.5]])
    bold_alpha = np.array([[[0.2, 0.0], [0.0, 0.2]],
                           [[0.0, 0.3], [0.3, 0.0]]])
    X, y = make_classification(bold_mu, bold_alpha, beta=2.0, end_time=300.0,
                               n_samples=4, n_classes=2, random_state=1)
    # max_iter=1: only prediction is exercised, convergence is irrelevant
    classifier = ERMCLassifier(decay=2.0, max_iter=1, verbose_bar=False, verbose=False)
    classifier.fit(X, y, end_time=300.0)
    proba = classifier.predict_proba(X, end_time=300.0)
    assert np.all(np.isfinite(proba))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_make_classification_checks_the_weights():
    """Check that make_classification accepts NumPy integers
    and rejects weights of the wrong length."""
    bold_mu = np.full((2, 1), 0.5)
    bold_alpha = np.full((2, 1, 1), 0.2)
    X, y = make_classification(bold_mu, bold_alpha, 1.0, 5.0, n_samples=4,
                               n_classes=np.int64(2), weights=[0.5, 0.5],
                               random_state=0)
    assert len(X) == 4
    with pytest.raises(ValueError, match="length of weights"):
        make_classification(bold_mu, bold_alpha, 1.0, 5.0, n_samples=4,
                            n_classes=2, weights=[1.0])
