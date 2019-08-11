import numpy as np
from scipy.sparse import vstack

class DistanceBasedWeight:

    def cosine_similarity(distance, multiplier=1.0):
        '''
        Cosine similarity with multiplier

        :param distance:
        :param multiplier:
        :return:
        '''
        return (1 - distance) * multiplier

    def inverse_formula(distancesnp):
        '''
        Formula suggested by "A Lazy Learning Approach for Building Classification Models"
        :return:
        '''

        distancesnp[distancesnp == 0.0] = 0.99999

        largest_distance = distancesnp[-1]
        drk = distancesnp / largest_distance
        distancias_norm = 1 / drk
        V = len(distancesnp) / distancias_norm.sum()
        return V * distancias_norm

    def sigmoid_function(x, L=2, k=10, x0=0.6, a=1.5):
        '''
        Sigmoid function to get the weights for a given SIMILARITY (normalized between 0 and 1)

        :param x:
        :param L:
        :param k:
        :param x0:
        :param a:
        :return:
        '''
        similarity = 1 - x
        similarity[similarity == 0] = 0.001
        similarity /= np.max(np.abs(similarity), axis=0)

        similarity[similarity == 0] = 0.001

        return L / (1 + a * (np.exp((-k * (similarity - x0)))))

    # based on "A New Distance-weighted k -nearest Neighbor Classifier"
    def dwknn_weights(distancesnp):
        if distancesnp[-1] == distancesnp[0]:
            return distancesnp
        return (distancesnp[-1] - distancesnp) / (distancesnp[-1] - distancesnp[0]) * (
                    distancesnp[-1] + distancesnp[0]) / (distancesnp[-1] + distancesnp)

    # based on "A New Distance-weighted k -nearest Neighbor Classifier"
    def wknn_weights(distancesnp):
        if distancesnp[-1] == distancesnp[0]:
            return distancesnp
        return (distancesnp[-1] - distancesnp) / (distancesnp[-1] - distancesnp[0])

    def oversample(X, y, weights, m=2.0):
        m = len(weights) * m
        new_weights = np.around((m * weights) / np.sum(weights)).astype(int)

        for index in range(0,len(weights)):
            instance = X[index]
            toadd = np.concatenate(([X], np.repeat(instance, new_weights[index], axis=0)))
            X = vstack(toadd)
            y = np.concatenate((y, np.repeat(y[index], new_weights[index], axis=0)))

        return X, y

    def define_weight(weight_func, y_train, neigh_dist_base, neigh_ind_base):
        '''

        :param y_train:
        :param neigh_dist_base:
        :param neigh_ind_base:
        :return:
        '''

        weights = None

        # Create a classifier with and without
        classes_unique = np.unique(y_train[neigh_ind_base])
        if len(classes_unique) > 1:
            if weight_func == 'cosine':
                weights = DistanceBasedWeight.cosine_similarity(distance=neigh_dist_base, multiplier=1.0)
            elif weight_func == 'cosine2':
                weights = DistanceBasedWeight.cosine_similarity(distance=neigh_dist_base, multiplier=2.0)
            elif weight_func == 'inverse':
                weights = DistanceBasedWeight.inverse_formula(neigh_dist_base)
            elif weight_func == 'sigmoid':
                weights = DistanceBasedWeight.sigmoid_function(neigh_dist_base)
            elif weight_func == 'wknn':
                weights = DistanceBasedWeight.wknn_weights(neigh_dist_base)
            elif weight_func == 'dwknn':
                weights = DistanceBasedWeight.dwknn_weights(neigh_dist_base)
            elif weight_func == 'none':
                weights = np.ones(len(neigh_dist_base))
        return weights
