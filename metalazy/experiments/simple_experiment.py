from utils.dataset_reader import DatasetReader
from classifiers.metalazy import MetaLazyClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import numpy as np
import argparse
import time

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, help='dataset name')
    parser.add_argument('-p', help='path to thedirectory with  libsvm files')

    args = parser.parse_args()
    dataset = args.d
    path = args.p

    dataset_reader = DatasetReader(path)

    fold = 0
    result = {'25':[], '200':[]}

    start = time.time()

    while dataset_reader.has_next():
        print('FOLD {}'.format(fold))

        # Load the regular data
        X_train, y_train, X_test, y_test = dataset_reader.get_next_fold()

        for N in [25,200]:
            print('{} Neighbours'.format(N))
            clf = MetaLazyClassifier(n_neighbors=N, select_features=False, weight_function='inverse')
            clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            #print(classification_report(y_pred=y_pred, y_true=y_test))
            result[str(N)].append(f1_score(y_true=y_test,y_pred=y_pred, average='macro'))

        fold = fold + 1

    print(result)
    for N in ['25', '200']:
        print('{}: {}'.format(N,np.mean(np.array(result[N]))))

    end = time.time()
    print(end - start)

if __name__ == "__main__":
    main()