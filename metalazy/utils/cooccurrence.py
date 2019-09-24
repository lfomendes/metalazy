import random as rd
import numpy as np
from scipy import sparse
import time


class Cooccurrence:
    '''
    Class with functions to get coocurrence features
    '''

    def get_pairs_replacement(data, indices, n=2, number_of_cooccurrences=10):

        #probabilities = np.copy(data)

        # get the probabilities based on data
        total = np.sum(data)
        probabilities = data / total

        pairs = []
        # for i in range(0,int(len(indices) / 2)):
        for i in range(0, number_of_cooccurrences):
            p = sorted(np.random.choice(indices, n, replace=False, p=probabilities))
            if p not in pairs:
                pairs.append(p)
        return pairs

    def get_pairs(data, indices, n=2):

        values = np.copy(data)

        # normalize values
        values /= np.max(np.abs(values), axis=0)

        choices = []
        # choose randomly but based on tfidf
        for value, indice in zip(values, indices):
            if rd.random() < value:
                choices.append(indice)
        # get pairs
        choices = np.array(choices)
        pairs = []
        while len(choices) > 1:
            index = np.random.choice(choices, n, replace=False)
            choices = choices[~np.in1d(choices, index)]
            pairs.append(index)
        return pairs

    def cooccurrence(X_train, instance, number_of_cooccurrences=10, seed_value=45):
        '''
        Create a new feature that is a linear combination of other features of the matrixes given.
        Example = new_feature = (0.12*feature_a) + (0.12*feature_b)
        '''

        feature_number = len(instance.indices)
        if feature_number < 2 or number_of_cooccurrences < 1:
            return X_train, instance

        start = time.time()
        indices = Cooccurrence.get_pairs_replacement(instance.data, instance.indices)
        end = time.time()
        #print('lala:{}'.format((end-start)*100))

        start = time.time()
        cols = [X_train]
        cols_test = [instance]
        for combination_i in range(0, len(indices)):
            features_to_combine = X_train[:, indices[combination_i]].toarray()
            idx_has_all_words = np.logical_and.reduce(features_to_combine, axis=1)[np.newaxis].T
            col = features_to_combine.mean(axis=1)[np.newaxis].T
            col *= idx_has_all_words

            col_test = sparse.csr_matrix(instance[:, indices[combination_i]].mean(axis=1))
            cols.append(col)
            cols_test.append(col_test)

        end = time.time()
        #print('lele:{}'.format((end-start)*100))

        start = time.time()
        X_train = sparse.hstack(cols, format='csr')
        instance = sparse.hstack(cols_test, format='csr')
        end = time.time()
        #print('lili:{}'.format((end-start)*100))
        #print()

        return X_train, instance
