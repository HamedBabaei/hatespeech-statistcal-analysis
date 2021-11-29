"""
    ML model utils
"""
import os
from .ngram_ml import NgramML
from .evaluation import evaluate
from datahandlers import DataReader, DataWriter
from pprint import pprint
import re
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score


def cross_validation(clf, X, y, cv=5):
    scorer = make_scorer(f1_score, average='macro')
    scores = cross_val_score(clf, X, y, cv=cv, scoring=scorer)
    print("CV RESULTS:", scores, scores.std(), scores.mean())
    return scores


def process_tweet(tweet):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ", tweet.lower()).split())


def ml_twitter_train_model(model: NgramML, config):
    print("TRAINING TFIDF + ML MODEL.....")
    train_data = eval("DataReader." + config.loader)(path=os.path.join(config.intermediate_train_dir, config.train_name))
    train_data['processed_tweets'] = train_data['tweet'].apply(process_tweet)
    print(f"Size of the train set is: {train_data.shape[0]}")
    X_train, y_train = train_data['processed_tweets'], train_data['label']

    print("5-fold cross validation .....")
    scores = cross_validation(model.get_model(), X_train, y_train)

    print("Training the model .....")
    model.fit(X_train, y_train)

    path_to_model = os.path.join(config.pre_trained_dir, config.model_name, config.model_name+".sav")
    print(f"Save pretrained model into :{path_to_model}")
    DataWriter.write_pkl(model, path_to_model)

    report = {
        "5-fold": [float(score) for score in list(scores)],
        "std": scores.std(),
        "mean": scores.mean(),
        "CI": scores.std()*2
    }
    path_to_report = os.path.join(config.logs_dir, config.dataset + "-evaluation-(5-CV)-" + config.model_name + ".json")
    print(f"Save cross validation results into:{path_to_report}")
    DataWriter.write_json(report, path_to_report)


def ml_twitter_test_model(model: NgramML, config):
    print(f"TEST TFIDF + ML MODEL FOR: {config.dataset.upper()}")

    val_data = eval("DataReader." + config.loader)(path=os.path.join(config.intermediate_train_dir, config.val_name))
    val_data['tweets'] = val_data['tweet'].apply(process_tweet)

    test_data = eval("DataReader." + config.loader)(path=os.path.join(config.intermediate_train_dir, config.test_name))
    test_data['tweets'] = test_data['tweet'].apply(process_tweet)

    print(f"Size of the test set is:  {test_data.shape[0]}")
    X_test, y_test = test_data['tweets'], test_data['label']
    X_val, y_val = val_data['tweets'], val_data['label']

    print("Making a predictions on test set")
    test_predict = model.predict(X_test)
    test_f1, test_acc, test_clf_report = evaluate(gold=y_test,
                                                   predicts=test_predict,
                                                   average='macro')

    val_predict = model.predict(X_val)
    val_f1, val_acc, val_clf_report = evaluate(gold=y_val,
                                                  predicts=val_predict,
                                                  average='macro')
    report = {
        "Test-F1 Macro": test_f1,
        "Test-accuracy": test_acc,
        "Test-classification-report": test_clf_report,
        "Test-gt": list(y_test),
        "Test-predict": [int(pred) for pred in list(test_predict)],
        "Val-F1 Macro": val_f1,
        "Val-accuracy": val_acc,
        "Val-classification-report": val_clf_report,
        "Val-gt": list(y_val),
        "Val-predict": [int(pred) for pred in list(val_predict)]
    }
    pprint(test_clf_report)

    pprint(val_clf_report)
    path_to_report = os.path.join(config.logs_dir, config.dataset+"-evaluation-"+config.model_name+".json")
    print(f"Save results with gt and predicts into :{path_to_report}")
    DataWriter.write_json(report, path_to_report)

