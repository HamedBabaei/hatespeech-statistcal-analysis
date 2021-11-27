"""
    LR model utils
"""
import os
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import  train_test_split
from pprint import pprint
from datahandlers import DataWriter, DataReader
from .roberta_ft import RoBERTaHSDetector
from .evaluation import evaluate
import transformers

def process_tweet(tweet):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ", tweet.lower()).split())


def roberta_twitter_train_model(config, device):
    print("transformers version:", transformers.__version__)
    def metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    print("TRAINING TFIDF + ML MODEL.....")

    train_data = eval("DataReader." + config.loader)(path=os.path.join(config.intermediate_train_dir, config.train_name))
    test_data = eval("DataReader." + config.loader)(path=os.path.join(config.intermediate_train_dir, config.test_name))
    train_data['tweets'] = train_data['tweet'].apply(process_tweet)
    test_data['tweets'] = test_data['tweet'].apply(process_tweet)
    train_data = train_data.rename(columns={"tweet": "text"})
    test_data = test_data.rename(columns={"tweet": "text"})
    print(f"Size of the train set is: {train_data.shape[0]}, "
          f"Number of hate speech samples:{sum(train_data['label'].tolist())}")
    print(f"Size of the test set is: {test_data.shape[0]}, "
          f"Number of hate speech samples:{sum(test_data['label'].tolist())}")
    print("STARTING THE FINE TUNING ROBERTA FOR DOWNSTREAM TASK")
    #model_path = os.path.join(config.roberta_fine_tune, config.checkpoint)
    roberta = RoBERTaHSDetector(model_path=config.roberta_base, tokenizer_path=config.roberta_base,
                                max_length=config.max_length, device=device)
    roberta.fine_tune(config, train_data, test_data, metrics)
    print("ENF OF THE FINE TUNING!")


def roberta_twitter_test_model(config, device):
    print(f"TEST RoBERTa MODEL FOR: {config.dataset.upper()}")
    test_data = eval("DataReader." + config.loader)(
        path=os.path.join(config.intermediate_train_dir, config.test_name))

    print(f"Size of the test set is:  {test_data.shape[0]}")
    test_data['tweets'] = test_data['tweet'].apply(process_tweet)
    X_test, y_test = test_data['tweets'].tolist(), test_data['label'].tolist()

    #model_path = os.path.join(config.roberta_fine_tune, config.checkpoint)
    model = RoBERTaHSDetector(model_path=config.roberta_fine_tune, tokenizer_path=config.roberta_fine_tune,
                              max_length=config.max_length, device=device)
    print("Making a predictions on test set")
    predicts = model.predict(X_test)
    f1, acc, clf_report = evaluate(gold=y_test,
                                   predicts=predicts,
                                   average='macro')
    report = {"F1 Macro": f1, "accuracy": acc, "classification-report": clf_report}

    pprint(report)

    path_to_report = os.path.join(config.logs_dir, config.dataset + "-evaluation-" + config.model_name + ".json")
    print(f"Save results with gt and predicts into :{path_to_report}")
    report["gt"], report['predict'] = list(y_test), [int(pred) for pred in list(predicts)]
    DataWriter.write_json(report, path_to_report)

