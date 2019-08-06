from metalazy.utils.dataset_reader import DatasetReader
from metalazy.classifiers.metalazy import MetaLazyClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import argparse
import time
import os
import random

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
    parser.add_argument('-j', help='number of jobs to run in parallel. use -1 for all - Default:-1')
    parser.add_argument('-g', help='Size of the sample to the hyperparameter search - Default-5000')

    args = parser.parse_args()

    output_path = args.o
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    path = args.p

    n_jobs = -1
    if args.j:
        n_jobs = int(args.j)

    grid_size = 5000
    if args.g:
        grid_size = int(args.g)

    dataset_reader = DatasetReader(path)

    fold = 0
    times = []

    specific_classifier = ['nb','logistic','extrarf']

    start = time.time()
    while dataset_reader.has_next():
        time_dic = {}
        print('FOLD {}'.format(fold))

        # Load the regular data
        X_train, y_train, X_test, y_test = dataset_reader.get_next_fold()

        result_df = pd.DataFrame()

        # for each fold we vary the specific classifier
        for specific in specific_classifier:
            print('Running for specific {}'.format(specific))

            # setting the 0 configurations (turn off cooc or weight)
            estimator =  MetaLazyClassifier( specific_classifier=specific,
                         select_features=False,
                         n_jobs=n_jobs,
                         number_of_cooccurrences=0,
                         weight_function='none',
                         grid_size=grid_size)

            print(estimator)

            # Fit the train data
            fit(estimator, X_train, y_train, time_dic)

            # Predict
            y_pred = predict(estimator, X_test, time_dic)

            # Save the result
            result_df[specific] = y_pred

        result_df['y_test'] = y_test

        times.append(time_dic)
        fold = fold + 1

        print(result_df.head(10))
        result_df.to_csv(output_path + '/result_oracle_off_fold_{}.csv'.format(fold), index=False)

        times_dataframe = pd.DataFrame(data=times)
        print(times_dataframe.head(10))
        times_dataframe.to_csv(output_path + '/times.csv', index=False)

    end = time.time()
    print('Total time: {}'.format((end - start)))

    times_dataframe = pd.DataFrame(data=times)
    print(times_dataframe.head(10))
    times_dataframe.to_csv(output_path + '/times.csv', index=False)


if __name__ == "__main__":
    main()
