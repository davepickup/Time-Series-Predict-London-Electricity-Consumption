import argparse
import os
import pickle
import urllib.request
import numpy as np
from sklearn.metrics import mean_absolute_error

import pandas as pd

from model import EnergyModel as Model


DATA_DIR = "data"
PICKLE_NAME = 'model.pickle'
TRAIN_NAME = "train.csv"
TEST_NAME = "test.csv"
TEST_TRUE = "y_test.csv"


def train_model():
    df = pd.read_csv(os.sep.join([DATA_DIR, TRAIN_NAME]))

    my_model = Model()
    X_train, y_train = my_model.preprocess_training_data(df)
    my_model.fit(X_train, y_train)

    # Save to pickle
    with open(PICKLE_NAME, 'wb') as f:
        pickle.dump(my_model, f)


def test_model():
    df = pd.read_csv(os.sep.join([DATA_DIR, TEST_NAME]))

    # Load pickle
    with open(PICKLE_NAME, 'rb') as f:
        my_model = pickle.load(f)

    X_test = my_model.preprocess_unseen_data(df)
    preds = my_model.predict(X_test)
    print("### Your predictions ###")
    print(preds)
    return preds

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true))	

def eval_model():
    df = pd.read_csv(os.sep.join([DATA_DIR, TEST_NAME]))

    # Load pickle
    with open(PICKLE_NAME, 'rb') as f:
        my_model = pickle.load(f)

    X_test = my_model.preprocess_unseen_data(df)
    preds = my_model.predict(X_test)
    true = pd.read_csv(os.sep.join([DATA_DIR, TEST_TRUE]))
    print("MAPE = {}".format(mape(true["consumption"], preds)))


def main():
    parser = argparse.ArgumentParser(
        description="A command line-tool to manage the project.")
    parser.add_argument(
        'stage',
        metavar='stage',
        type=str,
        choices=['train', 'test', 'eval'],
        help="Stage to run.")

    stage = parser.parse_args().stage

    if stage == "train":
        print("Training model...")
        train_model()

    elif stage == "test":
        print("Testing model...")
        test_model()

    elif stage == "eval":
        print("Evaluating model...")
        eval_model()


if __name__ == "__main__":
    main()
