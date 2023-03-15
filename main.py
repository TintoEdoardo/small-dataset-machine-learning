"""
EXPERIMENTS START
"""

import Dataset
import DataAnalysis


if __name__ == '__main__':

    #  Load Samples, Features and Labels
    samples, features, labels = Dataset.Load_Dataset ()

    #  Check for consistency and extract features
    features = Dataset.Check_Features_Consistency (features)

    #  Extract relevant samples
    excluded_features = DataAnalysis.Excluded_Features ()
    samples = DataAnalysis.Exclude_Features (samples, excluded_features)

    #  Compute different set of features
    hi_features      = DataAnalysis.Highly_Important_Features ()
    t_features       = DataAnalysis.Time_Features ()
    target_features  = DataAnalysis.Target_Features ()
    fr_features      = DataAnalysis.First_Round_Features ()
    sr_features      = DataAnalysis.Second_Round_Features ()
    tr_features      = DataAnalysis.Third_Round_Features ()

    """  Produce correlation diagrams  """
    feat_to_plot = [
        {
            'title'    : 'Highly important features',
            'name'     : 'hi_feat',
            'features' : hi_features + target_features
        },
        {
            'title'    : 'First round features',
            'name'     : 'fr_feat',
            'features' : fr_features + target_features
        },
        {
            'title'    : 'Second round features',
            'name'     : 'sr_feat',
            'features' : sr_features + target_features
        },
        {
            'title'    : 'Third round features',
            'name'     : 'tr_feat',
            'features' : tr_features + target_features
        },
        {
            'title'   : 'Highly important features (with time)',
            'name'    : 'hi_wTime_feat',
            'features': hi_features + t_features + target_features
        },
        {
            'title'   : 'First round features (with time)',
            'name'    : 'fr_wTime_feat',
            'features': fr_features + t_features + target_features
        },
        {
            'title'    : 'Second round features (with time)',
            'name'     : 'sr_wTime_feat',
            'features' : sr_features + t_features + target_features
        },
        {
            'title'    : 'Third round features (with time)',
            'name'     : 'tr_wTime_feat',
            'features' : tr_features + t_features + target_features
        }
    ]

    for ftp in feat_to_plot:
        f = ftp ['features']
        n = ftp ['name']
        t = ftp ['title']
        DataAnalysis.Build_Correlation_Matrix\
            (DataAnalysis.Select_Features (samples, f), f , n , t)


    """  Produce comparison diagrams  """
    for ftp in feat_to_plot:
        f = ftp['features']
        n = ftp['name']
        t = ftp['title']

        test_size = 0.5
        X_train, X_test, Y_train, Y_test = DataAnalysis.Split_Samples (samples, f, test_size)

        Y_list   = [Y_test]
        Y_labels = ["Measured MUs"]

        #  Iterates for each number of folding
        #  during validation
        for model in ['SVR', 'Tree', 'CNN']:

            #  Iterates for each model used
            for cv in [2, 4]:
                Y_current  = DataAnalysis.Compare_Model (X_train, X_test, Y_train, model, cv)
                Y_label    = "Predicted UM (" + model + ", " + "CV = " + str (cv) + ")"
                Y_list.append (Y_current)
                Y_labels.append (Y_label)

        file_name = "comp_" + n + ".png"
        DataAnalysis.Build_Comparison_Graphs (Y_list, Y_labels, t, file_name)

