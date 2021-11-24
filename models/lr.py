"""
    * TFIDF + LogisticRegression Model
    * BERT Model
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class NLR:
    """
        Modularized Implementation of TFIDF LogiscticRegression Model
    """
    def __init__(self, config):
        self.model = Pipeline([('TFIDF', TfidfVectorizer(ngram_range=config.ngram_range,
                                                         stop_words=config.stop_words,
                                                         sublinear_tf=config.sublinear_tf)), 
                               ('LogiscticRegression', LogiscticRegression(C=config.C))])
    
    def fit(self, X:list, y:list):
        self.model.fit(X, y)

    def predict(self, X:list) -> list:
        return self.model.predict(X)
    