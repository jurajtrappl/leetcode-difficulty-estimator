"""Building blocks for tuning the scikit-learn models (used by sklearn_pipeline.ipynb).

Two small wrappers make things that are normally fixed choices into hyperparameters that a
search can tune:

* WeightedClassifier.weight_power: how strongly rare classes are up-weighted.
      0   -> no weighting (the model tends to say "Medium" a lot)
      1   -> fully "balanced" weights (every class counts equally; can over-correct)
      0.5 -> halfway (square root of the balanced weights)
* OrdinalRegressor.low / .high: where a regression output is cut into Easy / Medium / Hard.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.pipeline import FeatureUnion
from sklearn.utils.class_weight import compute_sample_weight


def weights(y, power: float):
    """Balanced sample weights raised to `power` (0 = all ones, 1 = fully balanced)."""
    if power == 0:
        return np.ones(len(y))
    return compute_sample_weight("balanced", y) ** power


class WeightedClassifier(ClassifierMixin, BaseEstimator):
    """Any sklearn classifier, trained with class weights of tunable strength."""

    def __init__(self, estimator=None, weight_power: float = 1.0):
        self.estimator = estimator
        self.weight_power = weight_power

    def fit(self, X, y):
        y = np.asarray(y)
        self.estimator_ = clone(self.estimator).fit(X, y, sample_weight=weights(y, self.weight_power))
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


class OrdinalRegressor(ClassifierMixin, BaseEstimator):
    """Regress the label as a number (Easy 0, Medium 1, Hard 2), then cut it into classes:
    below `low` -> Easy, above `high` -> Hard, in between -> Medium."""

    def __init__(self, estimator=None, weight_power: float = 1.0, low: float = 0.75, high: float = 1.25):
        self.estimator = estimator
        self.weight_power = weight_power
        self.low = low
        self.high = high

    def fit(self, X, y):
        y = np.asarray(y)
        self.estimator_ = clone(self.estimator).fit(X, y.astype(float), sample_weight=weights(y, self.weight_power))
        self.classes_ = np.array([0, 1, 2])
        return self

    def predict(self, X):
        z = self.estimator_.predict(X)
        return np.where(z < self.low, 0, np.where(z > self.high, 2, 1))


def text_features(word_ngrams=(1, 3), char_ngrams=(1, 5), lowercase=True) -> FeatureUnion:
    """TF-IDF over word n-grams + binary TF-IDF over character n-grams (as in the original project)."""
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=word_ngrams, lowercase=lowercase, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=char_ngrams, lowercase=lowercase, binary=True)),
    ])


# Scorer for quadratic weighted kappa (Easy->Hard counts as a bigger mistake than Easy->Medium).
qwk_scorer = make_scorer(cohen_kappa_score, weights="quadratic")
