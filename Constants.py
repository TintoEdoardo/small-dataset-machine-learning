"""
CONSTANTS AND CONFIGURATION
"""

""" Constants"""
path_to_datasets       = "Datasets/"
path_to_graphs         = "Graphs/"
dataset_section_prefix = "section_"
last_section           = 3
dataset_sections       = range (1, last_section + 1)
dataset_section_ext    = ".csv"
dataset_section_name   = \
    [dataset_section_prefix + str (i) + dataset_section_ext for i in dataset_sections]

"""Highly Important Features"""
hi_features = [
    'Fat mass (g).1',
    '% FAT.1',
    'Lean (g)',
    'TBW%',
    'Rz',
    'Xc',
    'MVC (Kg)'
]

fr_features = [
    'Rz',
    'Xc',
    'TBW',
    'ECW',
    'PA',
    'TBW%',
    'ECW%',
    'ICW%',
    'BCMI',
    'ECM',
    'ASMM',
    'RZ (?)',
    'XC (?)',
    'PhA (°)',
    'LST (kg)'
]

sr_features = [
    'BMC (g)',
    'Fat mass (g)',
    'Lean Mass (g)',
    'Total mass (g)',
    '% FAT',
    'Est VAT mass (g)',
    'Est VAT volume (cm3)',
    'Est VAT area (cm2)',
    'BMC (g).1',
    'Fat mass (g).1',
    'Lean (g)',
    'Lean + BMC',
    'Total mass',
    '% FAT.1',
    'Area (cm2)',
    'BMC (g).2',
    'BMD (g/cm2)'
]

tr_features = [
    'MVC (Kg)',
    'MVC/lean mass total',
    'MVC/lean mass right leg'
]

we_features = ['Week']

ta_features = ['UM a prova']

ex_features = [
    'Rampa_15_1a',
    'Rampa_15_2a',
    'Rampa_15_3a',
    'Rampa_35_1a',
    'Rampa_35_2a',
    'Rampa_50_1a',
    'Rampa_50_2a',
    'Rampa_70_1a',
    'Rampa_70_2a'
]

"""Configuration data"""
in_debug_mode          = False
