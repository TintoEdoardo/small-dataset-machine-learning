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


def Compare_Model (samples, features, model, test_size, cv, name, title):
    """
    :param test_size: float
    :param cv: int
    :param name: str
    :param title: str
    :param samples: pandas DataFrame
    :param features: list (str)
    :param model: str
    :return: @show image
    """
    X = Select_Features (samples, features)
    Y = Select_Features (samples, Target_Features ()).values.ravel ()

    X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size=test_size, random_state=0)

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

    #  Plot the results
    plt.clf()
    plt.gcf().set_size_inches (12, 10)
    plt.plot (clf.predict (X_test), label="Predicted MUs")
    plt.plot (Y_test, label="Measured MUs")
    plt.legend ()
    plt.title (title)
    exp_tag = "_" + model + "_ts" + str (test_size) + "_cv" + str (cv)
    plt.savefig\
        (Constants.path_to_graphs + "comp_" + name + exp_tag + ".png", bbox_inches='tight')



