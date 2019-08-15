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
    result = []
    times = []

    start = time.time()
    while dataset_reader.has_next():
        time_dic = {}
        print('FOLD {}'.format(fold))
        start_fold = time.time()

        # Load the regular data
        X_train, y_train, X_test, y_test = dataset_reader.get_next_fold()

        # Create the classifier
        clf = MetaLazyClassifier(select_features=False,
                                 n_jobs=n_jobs,
                                 grid_size=grid_size)

        # tuned_parameters = [{'specific_classifier': ['nb'],
        #                      'weight_function': ['none', 'cosine', 'inverse'],
        #                      'n_neighbors': [200, 50], 'number_of_cooccurrences': [1, 10]}]

        #tuned_parameters = [{'specific_classifier': ['nb', 'logistic', 'extrarf'],
        tuned_parameters = [{'specific_classifier': ['nb', 'logistic', 'extrarf'],
                             'weight_function': ['cosine', 'inverse'],
                              'n_neighbors': [100], 'number_of_cooccurrences': [10]}]

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

        print('Macro: {}'.format(f1_score(y_true=y_test, y_pred=y_pred, average='macro')))
        print('Micro: {}'.format(f1_score(y_true=y_test, y_pred=y_pred, average='micro')))
        times.append(time_dic)
        fold = fold + 1
        end_fold = time.time()
        print('Total fold time: {}'.format((end_fold - start_fold)))
        print('train size {}'.format(X_train.shape))
        print('test size {}'.format(X_test.shape))
        print()

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
