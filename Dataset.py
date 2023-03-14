"""
DATASET MANAGEMENT AND VALIDATION FUNCTIONS
"""

import Constants
import pandas


def Load_Dataset ():
    """
    :return: samples, features: list (dict), list (list (str))
    """

    samples  = pandas.DataFrame ()
    features = []

    #  The input dataset is divided in sections,
    #  each sections stored in a dedicated file.
    #  Here we iterate over each of this section
    for section in Constants.dataset_section_name:

        #  The current section content is read
        #  into 'file'
        path_to_section = Constants.path_to_datasets + section
        section_file    = open (path_to_section, newline='')

        #  The 'section_file' is read into a pandas
        #  DataFrame, and the header of the dataset
        #  (feature names) is acquired
        section_content = pandas.read_csv (section_file)
        header          = section_content.columns.to_list ()

        #  Debug mode
        if Constants.in_debug_mode:
            print (header)

        #  Debug mode
        if Constants.in_debug_mode:
            print ("Data from file " + section + ": ")
            print (section_content)

        samples = pandas.concat ([samples, section_content])
        features.append (header)

    #  Extract labels
    lables  = samples ['Patient']
    samples = samples.drop ('Patient', axis=1)

    return samples, features, lables


def Check_Features_Consistency (features):
    """
    :param: features: list (list (str))
    :return: features: list (str)
    """
    base_features     = features [0]
    features_len      = len (base_features)
    erroneous_feature = []

    #  Check if all the features extracted from
    #  the dataset sections are consistent (identical
    #  among each other)
    current_list = 0
    for f_list in features [1:]:
        current_list += 1

        #  Compare each feature between two lists
        for i in range (0, features_len):

            #  Store the different features between
            #  two dataset sections
            if f_list [i] != base_features [i]:
                erroneous_feature.append ("[ list_" + str (current_list) + " [" + f_list [i] + "], " +
                                          "list_" + str (0) + " [" + base_features [i] + "] ]")

    #  Check if the list of erroneous features
    #  is empty; if so, print the list
    if len (erroneous_feature) > 0:
        print ("Error: the following features are different: ")
        for e in erroneous_feature:
            print (e)

    #  Otherwise the execution ends returning 0
    return base_features [1:]
