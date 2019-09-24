import numpy as np
from random import *
from scipy import sparse
import numpy as np

def example():
    a = np.arange(12.0) - 6.0
    a.shape = 3, 4
    a = sparse.csr_matrix(a)

    print(a.toarray())
    print()
    print(np.logical_and.reduce(a, axis=0).toarray())
    print()
    print(np.multiply.reduce(a, axis=0).toarray())

def lala():
    X_train = np.arange(12.0) - 6.0
    X_train.shape = 3, 4
    X_train = sparse.csr_matrix(X_train)
    instance = np.array([1.0,2.0,3.0,4.0])
    indices = [[0,1], [0,2], [0,3]]
    print(X_train)

    cols = [X_train]
    cols_test = [instance]
    for combination_i in range(0, len(indices)):
        # for each matrix we create a new column
        features_to_combine = X_train[:, indices[combination_i]].toarray()
        idx_has_all_words = np.logical_and.reduce(features_to_combine, axis=0)
        col = features_to_combine.mean(axis=1)[np.newaxis].T
        col *= idx_has_all_words

        col_test = sparse.csr_matrix(instance[:, indices[combination_i]].mean(axis=1))
        cols.append(col.toarray())
#example()
lala()
