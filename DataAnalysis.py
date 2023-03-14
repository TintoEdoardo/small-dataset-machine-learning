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


"""
#  PREVIOUS IMPLEMENTATION
def Select_Samples (samples, percentage):
    rows_number     = int (samples.shape [0] / 3)
    training_rows   = int (rows_number * percentage)
    validation_rows = rows_number - training_rows
    training_samples   = []
    validation_samples = []
    for i in range (3):
        for i_1 in range (training_rows):
            line_i = rows_number * i + i_1
            training_samples.append (line_i)
        for i_2 in range (validation_rows):
            line_i = rows_number * i + training_rows + i_2
            validation_samples.append (line_i)

    return samples.iloc [training_samples], samples.iloc [validation_samples]
"""


def Build_Correlation_Matrix (samples, features):
    """
    :param samples: pandas DataFrame
    :param features: list (str)
    :return: @show image
    """
    plt.matshow (samples.corr ())
    plt.xticks (range (len (features)), features, rotation=90)
    plt.yticks (range (len (features)), features)
    plt.colorbar ()
    plt.show ()


def Compare_Model (samples, features, model):
    """
    :param samples: pandas DataFrame
    :param features: list (str)
    :param model: str
    :return: @show image
    """
    X = Select_Features (samples, features)
    Y = Select_Features (samples, Target_Features ()).values.ravel ()

    X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size=0.5, random_state=0)

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
    clf   = GridSearchCV (estim, prams)
    clf.fit (X_train, Y_train)

    #  Plot the results
    plt.clf()
    plt.plot(clf.predict (X_test), label="predicted UM")
    plt.plot(Y_test, label="actual UM")
    plt.legend()
    plt.show()


"""
#  PREVIOUS IMPLEMENTATION
def Train_And_Validate (tr_samples, val_samples, features, model):

    # https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation

    target_vector = Select_Features (tr_samples, Target_Features ()).values.ravel ()
    tr_samples    = Select_Features (tr_samples, features)
    val_vector    = Select_Features (val_samples, Target_Features ()).values.ravel ()
    val_samples   = Select_Features (val_samples, features)

    if Constants.in_debug_mode:
        print (target_vector)

    if model == 'LinearSVR':
        clf = make_pipeline(StandardScaler(), svm.SVR(C=1))
    elif model == 'tree':
        clf = tree.DecisionTreeRegressor ()
    elif model == 'HistGradient':
        clf = HistGradientBoostingRegressor ()
    elif model == 'CNN':
        clf = MLPRegressor (solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    elif model == 'PCA':
        clf = PCA (n_components=2)
        X_r = clf.fit (tr_samples).transform (tr_samples)
        print (X_r)
        plt.figure()
        colors = ["navy", "turquoise", "darkorange"]
        lw = 2
        for color, i, target_name in zip(colors, range(12), ['Fat mass (g).1',
        '% FAT.1',
        'Lean (g)']):
            plt.scatter(
                X_r[target_vector == i, 0], X_r[target_vector == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
            )
        plt.legend(loc="best", shadow=False, scatterpoints=1)
        plt.title("PCA of IRIS dataset")
        plt.show()
    else:
        clf = -1

    clf.fit (tr_samples, target_vector)
    #  score   = clf.score (tr_samples, target_vector)
    #  weights = clf.coef_

    # score = cross_val_score (clf, tr_samples, target_vector, cv=2)
    print (val_vector)
    print (target_vector)
    print (clf.predict (val_samples))

    #  Validation
    plt.clf ()
    # plt.plot (clf.predict (val_samples), label="predicted UM")
    plt.plot(cross_val_predict (clf, val_samples, val_vector, cv=4), label="predicted UM")
    plt.plot (val_vector, label="actual UM")
    plt.legend ()
    plt.show ()
"""

