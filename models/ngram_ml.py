"""
    * TFIDF + LinearSVC Model
    * BERT Model
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


class NgramML:
    """
        Modularized Implementation of TFIDF + LinearSVC Model
    """
    def __init__(self, config):
        self.model = Pipeline([('TFIDF', TfidfVectorizer(ngram_range=config.ngram_range,
                                                         stop_words=config.stop_words,
                                                         sublinear_tf=config.sublinear_tf,
                                                         analyzer=config.analyzer)),
                               ("ML",  CalibratedClassifierCV(LinearSVC()))])

    def fit(self, X: list, y: list):
        self.model.fit(X, y)

    def predict(self, X: list) -> list:
        return self.model.predict(X)

    def get_model(self):
        return self.model
    