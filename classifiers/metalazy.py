import random
import numpy as np
from scipy import sparse

from sklearn.base import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import NearestNeighbors


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

    # The grid params for each weaker classifier
    weaker_grid_params = {
        'extrarf': [{'n_jobs': [-1], 'criterion': ['gini', 'entropy'], 'max_features': ['sqrt', 0.3, 'log2'],
                     'n_estimators': [100, 200]}],
        'nb': {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
        'logistic': [{'penalty': ['l1', 'l2'], 'class_weight': ['balanced', None],
                      'solver': ['liblinear'], 'C': [0.1, 1.0, 10, 20]},
                     {'solver': ['lbfgs'], 'C': [1, 10, 0.1, 0.01], 'n_jobs': [-1],
                      'class_weight': ['balanced', None],
                      'multi_class': ['ovr', 'multinomial']}]
    }

    def __init__(self, specific_classifier=None, n_jobs=-1, n_neighbors=200, metric='cosine', grid_size=1000,
                 random_state=42):
        """
        TODO
        """
        self.specific_classifier = specific_classifier
        self.grid_size = grid_size

        # everyone's params
        self.n_jobs = n_jobs
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

    def set_classifier(self, name):
        '''
        Create the classifier based on the specific_classifier parameter
        Possible values = ['rf', 'nb', 'extrarf', ''logistic]
        '''

        clf = None

        if name == 'rf':
            print('rf')
            self.weaker = RandomForestClassifier(random_state=self.random_state)
        elif name == 'nb':
            print('nb')
            self.weaker = MultinomialNB()
        elif name == 'extrarf':
            print('extrarf')
            self.weaker = ExtraTreesClassifier(random_state=self.random_state)
        elif name == 'logistic':
            print('logistic')
            self.weaker = LogisticRegression(random_state=self.random_state)
        else:
            raise Exception('The specific_classifier {} is not valid, use rf, nb, extrarf or logistic'.format(
                self.specific_classifier))

        self.weaker.set_params(**self.weaker_default_params[name])

    def get_sample_indices(self, X_train, size=1000):
        max_value = X_train.shape[0]
        if X_train.shape[0] < size:
            size = max_value
        return np.array(random.sample(range(max_value), size))

    def find_best_weaker_classifier(self, X_train, y_train, score='f1_macro'):
        # limit the dataset to make this phase faster
        # for each classifier
        # make the best param search
        # compare the best value
        # set the classifier

        inds = self.get_sample_indices(X_train, size=self.grid_size)
        X_train_filtered = X_train[inds]
        y_train_filtered = y_train[inds]

        best_score = 0.0
        best_clf = None
        for clf_name in self.possible_weakers:
            print('Testing result with {}'.format(clf_name))
            self.set_classifier(clf_name)

            tuned_parameters = self.weaker_grid_params[clf_name]

            grid = GridSearchCV(self.weaker, tuned_parameters, cv=3, scoring=score)
            grid.fit(X_train_filtered, y_train_filtered)

            print('Score: {}'.format(grid.best_score_))
            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                best_clf = grid.best_estimator_

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

        # for each test instance
        for i, ids in enumerate(idx):

            # Filter the X_train with the neighbours
            X_t = self.X_train[ids]

            # Create a specific classifier for this instance
            weaker_aux = clone(self.weaker)

            # TODO Create co-occorrence features

            # TODO Find weights

            # only fit the classifier if there is more than 1 class on the neighbourhood
            if len(np.unique(self.y_train[ids])) > 1:
                # fit the classifier
                weaker_aux.fit(X_t, self.y_train[ids], (1 - dists[i]))
                pred[i, np.searchsorted(self.classes_, weaker_aux.classes_)] = weaker_aux.predict_proba(X[[i]])[0]
            else:
                pred[i] = np.zeros((1, self.n_classes_))
                pred[i][self.y_train[ids][0]] = 1.0

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
