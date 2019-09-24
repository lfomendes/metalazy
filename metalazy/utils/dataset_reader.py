import os
from sklearn.datasets import load_svmlight_file, load_svmlight_files


class DatasetReader:
    '''
        Used to read the dataset and return the train and test instances
        It can be used to read from a whole file or
        using the LBD default fold partition
    '''

    def __init__(self, path):
        self.current_fold = 0
        self.default_fold_partition = True
        self.folds_path = path
        self.split = []

        print('Loading folds')
        self.train_folds = sorted(
            [filename for filename in os.listdir(self.folds_path) if
             (filename.startswith("train") or filename.startswith("treino"))])
        self.test_folds = sorted(
            [filename for filename in os.listdir(self.folds_path) if filename.startswith("test")])

    def _load_dataset_from_folds(self, train_file, test_file):
        print(train_file)
        print(test_file)
        result = load_svmlight_files([train_file, test_file])
        X_train = result[0]
        y_train = result[1]
        X_test = result[2]
        y_test = result[3]

        return X_train, y_train, X_test, y_test

    def get_next_fold(self):

        # If it is the default partition
        if self.default_fold_partition:
            if self.current_fold < len(self.train_folds):
                train_fold = self.train_folds[self.current_fold]
                test_fold = self.test_folds[self.current_fold]

                self.current_fold = self.current_fold + 1

                return self._load_dataset_from_folds(os.path.join(self.folds_path, train_fold),
                                                     os.path.join(self.folds_path, test_fold))
            else:
                return None
        else:
            if self.current_fold < len(self.split):
                train_index = self.split[self.current_fold][0]
                test_index = self.split[self.current_fold][1]

                # Returning the next fold from the full libsvm file
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]

                self.current_fold = self.current_fold + 1
                return X_train, y_train, X_test, y_test
            else:
                return None

        self.current_fold += 1

    def has_next(self):
        # If it is the default partition
        if self.default_fold_partition:
            if self.current_fold < len(self.train_folds):
                return True
            else:
                return False
        else:
            if self.current_fold < len(self.split):
                return True
            else:
                return False