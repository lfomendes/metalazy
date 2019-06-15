from metalazy.classifiers.metalazy import MetaLazyClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore", message="Numerical issues were encountered ")

iris = datasets.load_iris()
X = iris.data
y = iris.target

# divide into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

clf = MetaLazyClassifier(n_neighbors=1)

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

knn = KNeighborsClassifier(n_neighbors=25)

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

print(classification_report(y_test, y_pred))

