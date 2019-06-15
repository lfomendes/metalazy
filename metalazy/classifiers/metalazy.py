import random
import numpy as np
from scipy import sparse
import time
import os

from sklearn.base import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import NearestNeighbors
from metalazy.utils.distanced_based_weights import DistanceBasedWeight
from sklearn.feature_selection import SelectPercentile, chi2
import multiprocessing as mp


class MetaLazyClassifier(BaseEstimator, ClassifierMixin):
    # The default params for each weaker classifier
    weaker_default_params = {
        'rf': {'class_weight': 'balanced', 'criterion': 'gini', 'max_features': 'sqrt', 'n_estimators': 200},
        'nb': {'alpha': 10},
        'extrarf': {'class_weight': 'balanced', 'criterion': 'gini', 'max_features': 'sqrt', 'n_estimators': 200},
        'logistic': {'penalty': 'l2', 'class_weight': 'balanced', 'solver': 'liblinear', 'C': 10}
    }

    # Classifiers to test for each dataset
    possible_weakers = ['nb', 'logistic', 'extrarf']
    #possible_weakers = ['logistic']

    # The grid params for each weaker classifier
    weaker_grid_params = {
        'extrarf': [{'criterion': ['gini', 'entropy'], 'max_features': ['sqrt', 0.3, 'log2'],
                     'n_estimators': [100, 200]}],
        'nb': {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
        'logistic': [{'penalty': ['l1', 'l2'], 'class_weight': ['balanced', None],
                      'solver': ['liblinear'], 'C': [0.1, 1.0, 10, 20]},
                     # {'solver': ['lbfgs'], 'C': [1, 10, 0.1, 0.01], 'n_jobs': [-1],
                     #  'class_weight': ['balanced', None],
                     #  'multi_class': ['ovr', 'multinomial']}
                     ]
    }

    def __init__(self, specific_classifier=None, n_jobs=-1, n_neighbors=200, metric='cosine',
                 grid_size=1000, weight_function='inverse', select_features=False, random_state=42):
        """
        TODO
        """
        self.specific_classifier = specific_classifier
        self.grid_size = grid_size
        self.weight_function = weight_function
        self.select_features = select_features

        # everyone's params
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        self.random_state = random_state

        # kNN params
        self.n_neighbors = n_neighbors
        self.metric = metric

        # params that will be defined later
        self.weaker = None
        self.kNN = None
        self.X_train = None
        self.y_train = None
        self.classes_ = None
        self.n_classes_ = None

        # setting the seed
        random.seed(random_state)

    def set_classifier(self, name, specif_jobs=1):
        '''
        Create the classifier based on the specific_classifier parameter
        Possible values = ['rf', 'nb', 'extrarf', ''logistic]
        '''

        clf = None

        if name == 'rf':
            print('rf')
            clf = RandomForestClassifier(random_state=self.random_state, n_jobs=specif_jobs)
        elif name == 'nb':
            print('nb')
            clf = MultinomialNB()
        elif name == 'extrarf':
            print('extrarf')
            clf = ExtraTreesClassifier(random_state=self.random_state, n_jobs=specif_jobs)
        elif name == 'logistic':
            print('logistic')
            clf = LogisticRegression(random_state=self.random_state, n_jobs=specif_jobs)
        else:
            raise Exception('The specific_classifier {} is not valid, use rf, nb, extrarf or logistic'.format(
                self.specific_classifier))

        clf.set_params(**self.weaker_default_params[name])

        return clf

    def get_sample_indices(self, X_train, size=1000):
        max_value = X_train.shape[0]
        if X_train.shape[0] < size:
            size = max_value
        return np.array(random.sample(range(max_value), size))

    def avaliate_weaker(self, clf_name, X, y, score='f1_macro'):
        '''
        Function to be called by each process (parallel).
        It finds the best parameter for this specific classifier
        :param clf_name:
        :param X:
        :param y:
        :param score:
        :return:
        '''
        print('MP:{}'.format(clf_name))
        weaker = self.set_classifier(clf_name)
        tuned_parameters = self.weaker_grid_params[clf_name]

        if clf_name == 'extrarf' and self.n_jobs > len(self.possible_weakers):
            weaker.n_jobs = self.n_jobs - len(self.possible_weakers) + 1

        start_grid = time.time()
        grid = GridSearchCV(weaker, tuned_parameters, cv=3, scoring=score)
        grid.fit(X, y)
        end = time.time()
        print('{} Total grid time: {}'.format(clf_name,(end-start_grid)))
        return grid.best_score_, grid.best_estimator_

    def find_best_weaker_classifier(self, X_train, y_train, score='f1_macro'):
        # limit the dataset to make this phase faster
        # for each classifier
        # make the best param search
        # compare the best value
        # set the classifier

        inds = self.get_sample_indices(X_train, size=self.grid_size)
        X_train_filtered = X_train[inds]
        y_train_filtered = y_train[inds]

        print('Starting parallel process: {}'.format(self.n_jobs))
        # Creating arguments to parallel evaluation
        args = []
        for x in self.possible_weakers:
            args.append((x, X_train_filtered, y_train_filtered, score))

        # TODO It is not totally parallel yet, the classifier or gridsearch may use more process
        pool = mp.Pool(processes=self.n_jobs)
        results = pool.starmap(self.avaliate_weaker,args)
        print(results)

        best_score = 0.0
        best_clf = None
        for result in results:
            value = result[0]
            estimator = result[1]

            print('Score: {}'.format(value))
            if value > best_score:
                best_score = value
                best_clf = estimator

        print('Best Classifier: {}\n\n'.format(best_clf))
        self.weaker = best_clf

    def fit(self, X, y=None):
        """
        TODO
        """
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        self.X_train = sparse.csr_matrix(X)
        self.y_train = np.array(y)

        # Fit the kNN classifier
        self.kNN = NearestNeighbors(n_jobs=self.n_jobs, n_neighbors=self.n_neighbors, algorithm='brute',
                                    metric=self.metric)
        self.kNN.fit(self.X_train, self.y_train)

        if self.specific_classifier:
            # choose the weaker classifier
            self.set_classifier(name=self.specific_classifier)
        else:
            # test which classifier is the best for this specific dataset
            self.find_best_weaker_classifier(X, y)

        return self

    def lazy_feature_selection(self, X, y, instance):
        selector = SelectPercentile(chi2, percentile=25).fit(X, y)
        new_X_train = selector.transform(X)
        instance_comb = selector.transform(instance)

        return new_X_train, instance_comb

    # TODO se coloca o X no self é melhor em termos de memoria?
    def predict_parallel(self, row, dists, X, pred):

        i, ids = row

        # Filter the X_train with the neighbours
        X_t = self.X_train[ids]
        instance = X[[i]].copy()

        # Select features
        if self.select_features:
            X_t, instance = self.lazy_feature_selection(X_t, self.y_train[ids], X[i])

        # Create a specific classifier for this instance
        weaker_aux = clone(self.weaker)

        # TODO Create co-occorrence features

        # Find weights
        weights = DistanceBasedWeight.define_weight(self.weight_function, self.y_train,
                                                    dists[i], ids)

        # only fit the classifier if there is more than 1 class on the neighbourhood
        if len(np.unique(self.y_train[ids])) > 1:
            # fit the classifier
            weaker_aux.fit(X_t, self.y_train[ids], weights)
            pred[i, np.searchsorted(self.classes_, weaker_aux.classes_)] = weaker_aux.predict_proba(instance)[0]
        else:
            pred[i] = np.zeros((1, self.n_classes_))
            pred[i][int(self.y_train[ids][0])] = 1.0

    def predict_proba(self, X):

        """Predict class for X.
        The predicted class of an input sample is computed as the majority
        prediction of the trees in the forest.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes.
        """

        # get knn for all test sample
        # TODO use fastKNN
        dists, idx = self.kNN.kneighbors(X, return_distance=True)

        # Start the prediction with zeros
        pred = np.zeros((len(idx), self.n_classes_))

        print('Starting the prediction in parallel {}'.format(self.n_jobs))
        pool = mp.Pool(self.n_jobs)
        [pool.apply(self.predict_parallel, args=(row, dists, X, pred)) for row in enumerate(idx)]
        #
        # # for each test instance
        # for i, ids in enumerate(idx):
        #
        #     if i % 500 == 0:
        #         print('Predicted {} instances'.format(i))
        #
        #     # Filter the X_train with the neighbours
        #     X_t = self.X_train[ids]
        #     instance = X[[i]].copy()
        #
        #     # TODO Chamar funcao paralela aqui
        #
        #     # Select features
        #     if self.select_features:
        #         X_t, instance = self.lazy_feature_selection(X_t, self.y_train[ids], X[i])
        #
        #     # Create a specific classifier for this instance
        #     weaker_aux = clone(self.weaker)
        #
        #     # TODO Create co-occorrence features
        #
        #     # Find weights
        #     weights = DistanceBasedWeight.define_weight(self.weight_function, self.y_train,
        #                                                 dists[i], ids)
        #
        #     # only fit the classifier if there is more than 1 class on the neighbourhood
        #     if len(np.unique(self.y_train[ids])) > 1:
        #         # fit the classifier
        #         weaker_aux.fit(X_t, self.y_train[ids], weights)
        #         pred[i, np.searchsorted(self.classes_, weaker_aux.classes_)] = weaker_aux.predict_proba(instance)[0]
        #     else:
        #         pred[i] = np.zeros((1, self.n_classes_))
        #         pred[i][int(self.y_train[ids][0])] = 1.0

        return pred

    def predict(self, X, y=None):
        '''
        TODO
        :param X:
        :param y:
        :return:
        '''
        pred = self.predict_proba(X)
        return self.classes_.take(np.argmax(pred, axis=1), axis=0)
