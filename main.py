"""

"""

import Dataset
import DataAnalysis


if __name__ == '__main__':

    #  Load Samples, Features and Label
    samples, features, labels = Dataset.Load_Dataset ()

    #  Check for consistency and extract features
    features = Dataset.Check_Features_Consistency (features)

    #  Extract relevant samples
    excluded_features = DataAnalysis.Excluded_Features ()
    rel_samples = DataAnalysis.Exclude_Features (samples, excluded_features)

    #  Obtain highly critical features and target features
    hi_features      = DataAnalysis.Highly_Important_Features ()
    target_features  = DataAnalysis.Target_Features ()
    first_r_features = DataAnalysis.First_Round_Features ()
    sec_r_features   = DataAnalysis.Second_Round_Features ()

    #  Display the correlation diagram
    # features_to_plot = hi_features + target_features
    features_to_plot = first_r_features + sec_r_features + target_features
    DataAnalysis.Build_Correlation_Matrix\
        (DataAnalysis.Select_Features (samples, features_to_plot), features_to_plot)

    DataAnalysis.Compare_Model (samples, first_r_features, "CNN")

    #  Plot first prediction test
    # DataAnalysis.Train_And_Validate (tr_samples_80, val_samples_80, first_r_features, 'LinearSVR')

    #  DataAnalysis.Build_Classification (samples, [i for i in range (0, len(labels))])

