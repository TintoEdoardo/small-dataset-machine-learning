"""
FUNCTIONS FOR MODEL TRAINING, VALIDATION AND TESTING
"""

import Constants

import matplotlib.pyplot     as plt
from sklearn.pipeline        import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn                 import svm, preprocessing, tree
from sklearn.preprocessing   import StandardScaler
from sklearn.neural_network  import MLPRegressor


def Highly_Important_Features ():
    """
    :return: list (str)
    """
    return Constants.hi_features


def First_Round_Features ():
    """
    :return: list (str)
    """
    return Constants.fr_features


def Second_Round_Features ():
    """
    :return: list (str)
    """
    return Constants.sr_features


def Third_Round_Features ():
    """
    :return: list (str)
    """
    return Constants.tr_features


def Time_Features ():
    """
    :return: list (str)
    """
    return Constants.we_features


def Target_Features ():
    """
    :return: list (str)
    """
    return Constants.ta_features


def Excluded_Features ():
    """
    :return: list (str)
    """
    return Constants.ex_features


def Select_Features (samples, features):
    """
    :param samples: pandas DataFrame
    :param features: list (str)
    :return: pandas DataFrame
    """
    return samples [features]


def Exclude_Features (samples, exc_features):
    """
    :param samples: pandas DataFrame
    :param exc_features: list (str)
    :return: pandas DataFrame
    """
    return samples.drop (exc_features, axis=1)


def Build_Correlation_Matrix (samples, features, name, title):
    """
    :param title: str
    :param name: str
    :param samples: pandas DataFrame
    :param features: list (str)
    :return: @show image
    """
    plt.matshow (samples.corr ())
    plt.xticks  (range (len (features)), features, rotation=90)
    plt.yticks  (range (len (features)), features)
    plt.title   (title)
    plt.colorbar ()
    plt.gcf ().set_size_inches (12, 10)
    plt.savefig\
        (Constants.path_to_graphs + "corr_mat_" + name + ".png", bbox_inches='tight')


def Split_Samples (samples, features, test_size):
    """
    :param samples: pandas DataFrame
    :param features: list (str)
    :param test_size: float
    :return: list (list (float))
    """
    X = Select_Features (samples, features)
    Y = Select_Features (samples, Target_Features()).values.ravel()

    return train_test_split (X, Y, test_size=test_size, random_state=0)


def Compare_Model (X_train, X_test, Y_train, model, cv):
    """
    :param X_train: list (list (float))
    :param X_test: list (list (float))
    :param Y_train: list (list (float))
    :param cv: int
    :param model: str
    :return: list (float)
    """

    #  Parameters varying on the model under test
    prams = -1
    mod   = -1
    prep  = -1

    if model == 'SVR':
        mod   = svm.SVR()
        prams = {'svr__C': [1, 10, 100, 1000]}
        prep  = StandardScaler()

    elif model == 'Tree':
        mod   = tree.DecisionTreeRegressor ()
        prams = {'decisiontreeregressor__criterion' : ['squared_error', 'absolute_error']}
        prep  = StandardScaler()

    elif model == 'CNN':
        mod   = MLPRegressor (solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        prams = {'mlpregressor__hidden_layer_sizes' : [(1,), (10,), (50,)],
                 'mlpregressor__solver' : ['lbfgs', 'sgd', 'adam'],
                 'mlpregressor__max_iter' : [10000]}
        prep  = preprocessing.MinMaxScaler ()

    #  Assemble preprocessor and estimator
    estim = make_pipeline (prep, mod)
    clf   = GridSearchCV (estim, prams, cv=cv)
    clf.fit (X_train, Y_train)

    return clf.predict (X_test)


def Build_Comparison_Graphs (Y_list, Y_labels, title, name):
    #  Plot the results
    plt.clf()
    plt.gcf().set_size_inches (12, 10)

    #  Add every array to the graph
    for i in range (len (Y_list)):
        if i == 0:
            plt.plot (Y_list [i], 'o-', label=Y_labels [i])
        else:
            plt.plot (Y_list [i], label=Y_labels [i])

    plt.legend ()
    plt.title (title)
    plt.savefig\
        (Constants.path_to_graphs + name + ".png", bbox_inches='tight')
