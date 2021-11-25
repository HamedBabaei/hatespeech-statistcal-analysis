"""
    LR model utils
"""
import os
from .ngram_ml import NgramML
from .evaluation import evaluate
from datahandlers import DataReader, DataWriter
from pprint import pprint
import re


def process_tweet(tweet):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ", tweet.lower()).split())


def twitter_train_model(model: NgramML, config):
    print("TRAINING TFIDF + ML MODEL.....")

    train_data = eval("DataReader." + config.loader)(path=os.path.join(config.intermediate_train_dir, config.train_name))
    train_data['processed_tweets'] = train_data['tweet'].apply(process_tweet)

    print(f"Size of the train set is: {train_data.shape[0]}")
    X_train, y_train = train_data['processed_tweets'], train_data['label']

    print("Training the model .....")
    model.fit(X_train, y_train)

    path_to_model = os.path.join(config.pre_trained_dir, config.model_name, config.model_name+".sav")
    print(f"Save pretrained model into :{path_to_model}")
    DataWriter.write_pkl(model, path_to_model)


def twitter_test_model(model: NgramML, config):
    print(f"TEST TFIDF + ML MODEL FOR: {config.dataset.upper()}")

    test_data = eval("DataReader." + config.loader)(path=os.path.join(config.intermediate_train_dir, config.test_name))
    test_data['processed_tweets'] = test_data['tweet'].apply(process_tweet)

    print(f"Size of the test set is:  {test_data.shape[0]}")
    X_test, y_test = test_data['processed_tweets'], test_data['label']

    print("Making a predictions on test set")
    predicts = model.predict(X_test)
    f1, acc, clf_report = evaluate(gold=y_test,
                                   predicts=predicts,
                                   average='macro')
    report = {"F1 Macro": f1, "accuracy": acc, "classification-report": clf_report}
    pprint(report)
    path_to_report = os.path.join(config.logs_dir, config.dataset+"-evaluation-"+config.model_name+".json")
    print(f"Save results with gt and predicts into :{path_to_report}")
    report["gt"], report['predict'] = list(y_test), [int(pred) for pred in list(predicts)]
    DataWriter.write_json(report, path_to_report)

