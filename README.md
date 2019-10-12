# MetaLazy

MetaLazy is a supervised algorithm that makes a lazy classification, in other words, MetaLazy only creates and fits a model after the test instance is given.

# Installation
After cloning the repository:

> git clone https://github.com/lfomendes/metalazy.git	

Setup your [virtualenv](https://virtualenv.pypa.io/en/latest/)  to install all requirements, you can follow this tutorial: 
[https://vitux.com/install-python3-on-ubuntu-and-set-up-a-virtual-programming-environment/](https://vitux.com/install-python3-on-ubuntu-and-set-up-a-virtual-programming-environment/)

Activate your environment:

> source env3/bin/activate

Install the requirements

> pip install -r requirements.txt

Run the metalazy example to check if everything went okay

> python metalazy/example/metalazy_example.py

# Example

To run most of the experiments you will need data in the libsvm format. And example of it can be found in the folder *examples/data/stanford_tweets_tfIdf_5fold*. This folder contains 10 gz files, each one is from a cross-validation divison in 5 folds, in other words, you will use train0.gz to train and will evaluate on test0.gz.


> python metalazy/experiments/simple_experiment.py -p metalazy/example/data/stanford_tweets_tfIdf_5fold/

# DatasetReader
The class DatasetReader makes it easy to iterate over each fold, you just have to create it passing the path to the folder with all files (following the stanford_tweets example) and use the has_next() and next() function:

> dataset_reader = DatasetReader(path)
> 
> while dataset_reader.has_next():
>
>        print('FOLD {}'.format(fold))
>
>        # Load the regular data
>        X_train, y_train, X_test, y_test = dataset_reader.get_next_fold()
