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

def get_estimator(specific, weight, cooc, oversampling):

    weight_value = 'inverse' if weight == 1 else 'none'
    cooc_value = 10 if cooc == 1 else 0

    # Create the classifier
    clf = MetaLazyClassifier(specific_classifier=specific,
                             select_features=False,
			                 n_neighbors=200,
                             weight_function=weight_value,
                             number_of_cooccurrences=cooc_value,
                             oversample=oversampling,
                             n_jobs=3)
    return clf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', help='path to the directory with  libsvm files')
    parser.add_argument('-o', help='path to the output directory')
    parser.add_argument('-j', help='number of jobs to run in parallel. use -1 for all - Default:-1')
    parser.add_argument('-g', help='Size of the sample to the hyperparameter search - Default-5000')
    parser.add_argument('-s', help='If should use oversampling or not')

    args = parser.parse_args()

    output_path = args.o
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    path = args.p
    oversampling = args.s == 'true'

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
    configurations = {'weight': [0, 1],
                      'cooccurrence': [0, 1]}

    start = time.time()
    while dataset_reader.has_next():
        time_dic = {}
        print('INITIATING FOLD {}'.format(fold))

        # Load the regular data
        X_train, y_train, X_test, y_test = dataset_reader.get_next_fold()

        result_df = pd.DataFrame()

        # for each fold we vary the specific classifier
        for specific in specific_classifier:
            for cooc in configurations['cooccurrence']:
                for weight in configurations['weight']:
                    print('FOLD {}'.format(fold))
                    print('Running for specific {} and cooc {} and weight {}'.format(specific, cooc, weight))

                    # setting the 0 configurations (turn off cooc or weight)
                    estimator = get_estimator(specific, weight, cooc, oversampling)

                    print(estimator)

                    # Fit the train data
                    fit(estimator, X_train, y_train, time_dic)

                    # Predict
                    y_pred = predict(estimator, X_test, time_dic)

                    # Save the result
                    result_df['{}_{}_{}'.format(specific, weight, cooc)] = y_pred

        result_df['y_test'] = y_test

        times.append(time_dic)
        fold = fold + 1

        print(result_df.head(10))
        result_df.to_csv(output_path + '/result_fatorial_oracle_fold_{}.csv'.format(fold), index=False)

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
