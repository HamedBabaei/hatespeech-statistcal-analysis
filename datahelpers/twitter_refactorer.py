"""
Creates processed dataset from twitter for the task
"""
import os
from pprint import pprint
from sklearn.model_selection import train_test_split
import pandas as pd

DATA_READER = None
DART_WRITER = None

def twitter(SPAN, config, data_reader, data_writer):
    """

        :param SPAN:
        :param config:
        :param data_reader:
        :param data_writer:
    """
    def get_stats(X):
        stats = {0: 0, 1: 0}
        for label in X:
            stats[label] += 1
        return stats

    global DATA_READER, DART_WRITER
    DATA_READER, DART_WRITER = data_reader, data_writer

    print("----------------------TWITTER DATA REFACTORIGN----------------------")
    print(f"{SPAN} WORKING ROOT DIR:{config.raw_train_dir}")

    print(f"{SPAN} LOADING train.csv")
    train_path = os.path.join(config.raw_train_dir, "train.csv")
    train_data = DATA_READER.load_csv(train_path)

    X, y = train_data['tweet'], train_data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    stats = {
        "all-data": get_stats(y), 
        "train": get_stats(y_train),
        "test": get_stats(y_test)
        }
    train_df = pd.DataFrame(data={"tweet":X_train, "label":y_train})
    test_df = pd.DataFrame(data={"tweet":X_test, "label":y_test})

    print(f"{SPAN} SAVING INTERMEIDATE DATA'S INTO DIR:{config.intermediate_train_dir}")
    
    train_data_path = os.path.join(config.intermediate_train_dir, "twitter_train.csv")
    print(f"\t\t SAVING CSV:: {train_data_path}")
    DART_WRITER.write_csv(data=train_df, path=train_data_path)
    
    test_data_path = os.path.join(config.intermediate_train_dir, "twitter_test.csv")
    print(f"\t\t SAVING CSV:: {test_data_path}")
    DART_WRITER.write_csv(data=test_df, path=test_data_path)
    
    stats_path = os.path.join(config.logs_dir, "twitter_stats.json")
    print(f"\t\t SAVING JSON:: {stats_path}")
    DART_WRITER.write_json(data=stats, path=stats_path)
