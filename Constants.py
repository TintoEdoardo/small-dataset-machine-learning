"""
CONSTANTS AND CONFIGURATION
"""

""" Constants"""
path_to_datasets       = "Datasets/"
dataset_section_prefix = "section_"
last_section           = 3
dataset_sections       = range (1, last_section + 1)
dataset_section_ext    = ".csv"
dataset_section_name   = \
    [dataset_section_prefix + str (i) + dataset_section_ext for i in dataset_sections]


"""Configuration data"""
in_debug_mode          = False
