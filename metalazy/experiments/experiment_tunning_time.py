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
    time_dic['total_pred'] = (end_pred - start_pred)
    print('Total pred time: {}'.format(time_dic['total_pred']))

    return y_pred


def choose_tunning_parameters(specific, weight, coccurrence):
    tuned_parameters = [{'n_neighbors': [200]}]

    classifiers = ['logistic', 'nb', 'extrarf']
    if coccurrence == 1:
        tuned_parameters[0].update({'number_of_cooccurrences': [0, 10]})
    if weight == 1:
        tuned_parameters[0].update({'weight_function': ['cosine', 'inverse']})
    if specific == 1:
         tuned_parameters[0].update({'specific_classifier': classifiers})
    # else:
    #     tuned_parameters[0].update({'specific_classifier': random.sample(classifiers, 1)})

    return tuned_parameters


def get_best_version_for_each_dataset(dataset, n_jobs, grid_size):
    '''
    Returns the MetaLazy with the best configuration found for each dataset

    :param dataset:
    :param n_jobs:
    :param grid_size:
    :return:
    '''
    if dataset in ['logistic_200_inverse','20ng', 'reut', 'reut90']:
        return MetaLazyClassifier(select_features=False, n_jobs=n_jobs, grid_size=grid_size,
                                  specific_classifier='logistic', weight_function='inverse', n_neighbors=200,
                                  number_of_cooccurrences=10)
    elif dataset in ['acm','logistic_200_cosine']:
        return MetaLazyClassifier(select_features=False, n_jobs=n_jobs, grid_size=grid_size,
                                  specific_classifier='logistic', weight_function='cosine', n_neighbors=200,
                                  number_of_cooccurrences=10)
    elif dataset in ['stanford', 'stanford_tweets', 'logistic_100_inverse']:
        return MetaLazyClassifier(select_features=False, n_jobs=n_jobs, grid_size=grid_size,
                                  specific_classifier='logistic', weight_function='inverse', n_neighbors=100,
                                  number_of_cooccurrences=10)
    elif dataset in ['4uni', 'webkb', 'extrarf_200_inverse']:
        return MetaLazyClassifier(select_features=False, n_jobs=n_jobs, grid_size=grid_size,
                                  specific_classifier='extrarf', weight_function='inverse', n_neighbors=350,
                                  number_of_cooccurrences=10)
    elif dataset in ['yelp', 'yelp_reviews', 'extrarf_200_cosine']:
        return MetaLazyClassifier(select_features=False, n_jobs=n_jobs, grid_size=grid_size,
                                  specific_classifier='extrarf', weight_function='cosine', n_neighbors=200,
                                  number_of_cooccurrences=10)
    elif dataset in ['nb_200_cosine']:
        return MetaLazyClassifier(select_features=False, n_jobs=n_jobs, grid_size=grid_size,
                                  specific_classifier='nb', weight_function='cosine', n_neighbors=200,
                                  number_of_cooccurrences=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', help='path to the directory with  libsvm files')
    parser.add_argument('-o', help='path to the output directory')
    parser.add_argument('-j', help='number of jobs to run in parallel. use -1 for all - Default:-1')
    parser.add_argument('-g', help='Size of the sample to the hyperparameter search - Default-5000')
    parser.add_argument('-d',
                        help='Use the dataset default parameters.Dont use this parameter if you want to do grid search')
    parser.add_argument('-t', help='Limit test size, for each fold only use this number of instances')

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

    dataset = None
    if args.d:
        dataset = args.d

    test_size_limit = None
    if args.t:
        test_size_limit = int(args.t)


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

        if test_size_limit:
            X_test = X_test[0:test_size_limit]
            y_test = y_test[0:test_size_limit]

        if dataset is None:
            # Create the classifier
            estimator = MetaLazyClassifier(select_features=False,
                                           n_jobs=n_jobs,
                                           grid_size=grid_size)

            tuned_parameters = choose_tunning_parameters(specific=1, weight=1, coccurrence=1)

            print(tuned_parameters)

            # first we find the best configuration in general
            print('GRID SEARCH FOR FOLD {}'.format(fold))
            start_grid = time.time()
            grid = GridSearchCV(estimator, tuned_parameters, cv=3, scoring='f1_macro', n_jobs=1)
            grid.fit(X_train, y_train)
            end = time.time()
            time_dic['grid'] = (end - start_grid)
            print('GENERAL - Total grid time: {}'.format((end - start_grid)))
            print('GENERAL - Best score was {} with \n {}'.format(grid.best_score_, grid.best_estimator_))

            estimator = grid.best_estimator_
            best_param = grid.best_params_
            print('GENERAL - Best param was {}\n'.format(grid.best_params_))
        else:
            print('Using default dataset parameters')
            estimator = get_best_version_for_each_dataset(dataset=dataset, n_jobs=n_jobs, grid_size=grid_size)
        print(estimator)

        estimator.log_time_file = output_path + '/log_times_{}.json'.format(fold)

        # Fit the train data
        fit(estimator, X_train, y_train, time_dic)

        # Predict
        y_pred = predict(estimator, X_test, time_dic)

        print('\nWeaker Classifier used:')
        print(str(estimator.weaker))
        # Save the result
        result.append({
            'macro': f1_score(y_true=y_test, y_pred=y_pred, average='macro'),
            'micro': f1_score(y_true=y_test, y_pred=y_pred, average='micro'),
            'config': str(estimator),
            'best_clf': str(estimator.weaker),
            'fold': str(fold),
        })

        print('Macro: {}'.format(f1_score(y_true=y_test, y_pred=y_pred, average='macro')))
        print('Micro: {}'.format(f1_score(y_true=y_test, y_pred=y_pred, average='micro')))
        times.append(time_dic)
        fold = fold + 1

        result_dataframe = pd.DataFrame(data=result)
        print(result_dataframe.head(10))
        result_dataframe.to_csv(output_path + '/result_tunning_time.csv', index=False)

        times_dataframe = pd.DataFrame(data=times)
        print(times_dataframe.head(10))
        times_dataframe.to_csv(output_path + '/times.csv', index=False)

        estimator.flush_log_time_file()

    print(result)

    end = time.time()
    print('Total time: {}'.format((end - start)))

    result_dataframe = pd.DataFrame(data=result)
    print(result_dataframe.head(10))
    result_dataframe.to_csv(output_path + '/result_tunning_time.csv', index=False)

    times_dataframe = pd.DataFrame(data=times)
    print(times_dataframe.head(10))
    times_dataframe.to_csv(output_path + '/times.csv', index=False)


if __name__ == "__main__":
    main()
