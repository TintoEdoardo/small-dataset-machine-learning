"""
DATASET MANAGEMENT AND VALIDATION FUNCTIONS
"""

import Constants
import csv


def Load_Dataset():
    """
    :return: samples, features: list (dict), list (list (str))
    """

    samples = []
    features = []

    #  The input dataset is divided in sections,
    #  each sections stored in a dedicated file.
    #  Here we iterate over each of this section
    for section in Constants.dataset_section_name:

        #  The current section content is read
        #  into 'file'
        path_to_section = Constants.path_to_datasets + section
        section_file = open(path_to_section, newline='')

        #  The header of the dataset (feature names)
        #  is acquired
        rows = csv.reader(section_file)
        header = next(rows)

        #  Debug mode
        if Constants.in_debug_mode:
            print(header)

        #  Finally turn the section into a list
        #  of dictionary
        section_content = csv.DictReader(section_file)
        samples_list = list(section_content)

        #  Debug mode
        if Constants.in_debug_mode:
            print("Data from file " + section + ": ")
            for i in samples_list:
                print(i)

        samples.append(samples_list)
        features.append(header)

    return samples, features


def Check_Features_Consistency(features):
    """
    :param features: list (list (str))
    :return: int
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
    return 0
