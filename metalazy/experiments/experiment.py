from metalazy.utils.dataset_reader import DatasetReader
from metalazy.classifiers.metalazy import MetaLazyClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import argparse
import time
import os


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


def fit(clf, X_train, y_train, time_dic):
    start_fit = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    time_dic['fit'] = (end - start_fit)
    print('Total fit time: {}'.format(time_dic['fit']))


def predict(clf, X_test, time_dic):
    start_pred = time.time()
    y_pred = clf.predict(X_test)
    end_pred = time.time()
    time_dic['pred'] = (end_pred - start_pred)
    print('Total pred time: {}'.format(time_dic['pred']))

    return y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', help='path to the directory with  libsvm files')
    parser.add_argument('-o', help='path to the output directory')
    parser.add_argument('-c', help='classifier')
    parser.add_argument('-k', help='number of neighbours')
    parser.add_argument('-w', help='weight function')

    args = parser.parse_args()

    output_path = args.o
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    classifier_name = args.c
    path = args.p
    k = int(args.k)
    weight_function = args.w

    dataset_reader = DatasetReader(path)

    fold = 0
    result = []
    times = []

    start = time.time()
    while dataset_reader.has_next():
        time_dic = {}
        print('FOLD {}'.format(fold))

        # Load the regular data
        X_train, y_train, X_test, y_test = dataset_reader.get_next_fold()

        # Create the classifier
        clf = MetaLazyClassifier(n_neighbors=k, select_features=False, weight_function=weight_function, n_jobs=4,
                                 grid_size=5000, number_of_cooccurrences=10)
        # clf = MultinomialNB(alpha=0.01)

        # Fit the train data
        fit(clf, X_train, y_train, time_dic)

        # Predict
        y_pred = predict(clf, X_test, time_dic)

        print(str(clf))
        print(str(clf.weaker))
        # Save the result
        result.append({
            'macro': f1_score(y_true=y_test, y_pred=y_pred, average='macro'),
            'micro': f1_score(y_true=y_test, y_pred=y_pred, average='micro'),
            'config': str(clf),
            'best_clf': str(clf.weaker),
        })

        print('Macro: {}'.format(f1_score(y_true=y_test, y_pred=y_pred, average='macro')))
        print('Micro: {}'.format(f1_score(y_true=y_test, y_pred=y_pred, average='micro')))
        times.append(time_dic)
        fold = fold + 1

    print(result)

    end = time.time()
    print('Total time: {}'.format((end - start)))

    result_dataframe = pd.DataFrame(data=result)
    print(result_dataframe.head(10))
    result_dataframe.to_csv(output_path + '/result.csv', index=False)

    times_dataframe = pd.DataFrame(data=times)
    print(times_dataframe.head(10))
    times_dataframe.to_csv(output_path + '/times.csv', index=False)


if __name__ == "__main__":
    main()
