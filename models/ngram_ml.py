"""
    * TFIDF + LogisticRegression Model
    * BERT Model
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


class NgramML:
    """
        Modularized Implementation of TFIDF Machine Learning Model
    """
    def __init__(self, config):
        self.model = Pipeline([('TFIDF', TfidfVectorizer(ngram_range=config.ngram_range,
                                                         stop_words=config.stop_words,
                                                         sublinear_tf=config.sublinear_tf)), 
                               ('ML', RandomForestClassifier(n_estimators=config.n_estimators))])

    def fit(self, X:list, y:list):
        self.model.fit(X, y)

    def predict(self, X: list) -> list:
        return self.model.predict(X)
    