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
        tuned_parameters[0].update({'number_of_cooccurrences': [5, 10, 20]})
    if weight == 1:
        tuned_parameters[0].update({'weight_function': ['cosine', 'inverse', 'dwknn']})
    if specific == 1:
        tuned_parameters[0].update({'specific_classifier': classifiers})
    else:
        tuned_parameters[0].update({'specific_classifier': random.sample(classifiers, 1)})

    return tuned_parameters


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
    result = []
    times = []

    configurations = {'specific_classifier': [0, 1],
                      'weight': [0, 1],
                      'cooccurrence': [0, 1]}

    start = time.time()
    while dataset_reader.has_next():
        time_dic = {}
        print('FOLD {}'.format(fold))

        # Load the regular data
        X_train, y_train, X_test, y_test = dataset_reader.get_next_fold()

        # Create the classifier
        clf = MetaLazyClassifier(select_features=False,
                                 n_jobs=n_jobs,
                                 grid_size=grid_size)

        # for each fold we vary weight function, number of co occurrences and the choosing of the classifier
        for specific in configurations['specific_classifier']:
            for weight in configurations['weight']:
                for cooccurrence in configurations['cooccurrence']:
                    print(
                        'Running for specific {}, weight {} and cooccurrence {}'.format(specific, weight, cooccurrence))

                    tuned_parameters = choose_tunning_parameters(specific=specific, weight=weight,
                                                                 coccurrence=cooccurrence)

                    print('GENERAL STARTING')
                    start_grid = time.time()
                    grid = GridSearchCV(clf, tuned_parameters, cv=3, scoring='f1_macro', n_jobs=1)
                    grid.fit(X_train, y_train)
                    end = time.time()
                    print('GENERAL - Total grid time: {}'.format((end - start_grid)))
                    print('GENERAL - Best score was {} with \n {}'.format(grid.best_score_, grid.best_estimator_))

                    grid.best_score_, grid.best_estimator_

                    # Fit the train data
                    fit(grid.best_estimator_, X_train, y_train, time_dic)

                    # Predict
                    y_pred = predict(grid.best_estimator_, X_test, time_dic)

                    print(str(grid.best_estimator_))
                    print(str(grid.best_estimator_.weaker))
                    # Save the result
                    result.append({
                        'macro': f1_score(y_true=y_test, y_pred=y_pred, average='macro'),
                        'micro': f1_score(y_true=y_test, y_pred=y_pred, average='micro'),
                        'config': str(grid.best_estimator_),
                        'best_clf': str(grid.best_estimator_.weaker),
                    })

                    configuration = {'weight': weight, 'specific': specific, 'cooc': cooccurrence}

                    result[-1].update(configuration)

        print('Macro: {}'.format(f1_score(y_true=y_test, y_pred=y_pred, average='macro')))
        print('Micro: {}'.format(f1_score(y_true=y_test, y_pred=y_pred, average='micro')))
        times.append(time_dic)
        fold = fold + 1

        result_dataframe = pd.DataFrame(data=result)
        print(result_dataframe.head(10))
        result_dataframe.to_csv(output_path + '/result_factorial.csv', index=False)

        times_dataframe = pd.DataFrame(data=times)
        print(times_dataframe.head(10))
        times_dataframe.to_csv(output_path + '/times.csv', index=False)

    print(result)

    end = time.time()
    print('Total time: {}'.format((end - start)))

    result_dataframe = pd.DataFrame(data=result)
    print(result_dataframe.head(10))
    result_dataframe.to_csv(output_path + '/result_factorial.csv', index=False)

    times_dataframe = pd.DataFrame(data=times)
    print(times_dataframe.head(10))
    times_dataframe.to_csv(output_path + '/times.csv', index=False)


if __name__ == "__main__":
    main()
