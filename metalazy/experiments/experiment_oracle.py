from metalazy.utils.dataset_reader import DatasetReader
from metalazy.classifiers.metalazy import MetaLazyClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
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


def choose_tunning_parameters(specific, weight, coccurrence):
    tuned_parameters = [{'n_neighbors': [100]}]

    classifiers = ['logistic', 'nb', 'extrarf']
    if coccurrence == 1:
        tuned_parameters[0].update({'number_of_cooccurrences': [10]})
    if weight == 1:
        tuned_parameters[0].update({'weight_function': ['cosine', 'inverse', 'dwknn']})
    if specific == 1:
        tuned_parameters[0].update({'specific_classifier': classifiers})
    else:
        tuned_parameters[0].update({'specific_classifier': random.sample(classifiers, 1)})

    return tuned_parameters

import copy

def get_best_estimator_for_specific_classifier(X_train,y_train, specific ,n_jobs, grid_size, fold):

    # Create the classifier
    clf = MetaLazyClassifier( specific_classifier=specific,
                             select_features=False,
                             n_jobs=n_jobs,
                             grid_size=grid_size)

    tuned_parameters = {'weight_function': ['cosine', 'inverse', 'dwknn'], 'number_of_cooccurrences': [5, 15]}

    # first we find the best configuration in general
    print('GRID SEARCH FOR FOLD {}'.format(fold))
    start_grid = time.time()
    grid = GridSearchCV(clf, tuned_parameters, cv=3, scoring='f1_macro', n_jobs=1)
    grid.fit(X_train, y_train)
    end = time.time()
    print('GENERAL - Total grid time: {}'.format((end - start_grid)))
    print('GENERAL - Best score was {} with \n {}'.format(grid.best_score_, grid.best_estimator_))

    estimator = grid.best_estimator_
    best_param = grid.best_params_

    print('GENERAL - Best param was {}\n'.format(grid.best_params_))

    return estimator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', help='path to the directory with  libsvm files')
    parser.add_argument('-o', help='path to the output directory')
    parser.add_argument('-s', help='if should be hyperparameter search for metalazy')
    parser.add_argument('-j', help='number of jobs to run in parallel. use -1 for all - Default:-1')
    parser.add_argument('-g', help='Size of the sample to the hyperparameter search - Default-5000')

    args = parser.parse_args()

    output_path = args.o
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    path = args.p

    search_params = args.s == 'true'

    n_jobs = -1
    if args.j:
        n_jobs = int(args.j)

    grid_size = 5000
    if args.g:
        grid_size = int(args.g)

    dataset_reader = DatasetReader(path)

    fold = 0
    result = []
    times = []

    specific_classifier = ['nb','logistic','extrarf']

    start = time.time()
    while dataset_reader.has_next():
        time_dic = {}
        print('FOLD {}'.format(fold))

        # Load the regular data
        X_train, y_train, X_test, y_test = dataset_reader.get_next_fold()

        result_df = pd.DataFrame()

        # for each fold we vary weight function, number of co occurrences and the choosing of the classifier
        for specific in specific_classifier:
            print('Running for specific {}'.format(specific))

            # setting the 0 configurations (turn off cooc or weight)
            if search_params:
                estimator = get_best_estimator_for_specific_classifier(X_train, y_train, specific ,n_jobs, grid_size, fold)
            else:
                estimator =  MetaLazyClassifier( specific_classifier=specific,
                             select_features=False,
                             n_jobs=n_jobs,
                             grid_size=grid_size)

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
        result_df.to_csv(output_path + '/result_oracle_fold_{}.csv'.format(fold), index=False)

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
