# PROGRAMMER: Carlos Mertens
# DATE CREATED: (DD/MM/YY) - 26/12/18
# REVISED DATE: (DD/MM/YY) - Not revise it yet
# PURPOSE: To employ several supervised algorithms to accurately model individuals' income 
#           using data collected from the 1994 U.S. Census. Choose the best candidate 
#           algorithm from preliminary results and optimize this algorithm to best model the 
#           data. Goal with this implementation is to construct a model that accurately predicts 
#           whether an individual makes more than $50,000.
#
# USAGE: This script requires Numpy, Pandas and Scikit_Learn to be installed within the Python environment. 
# 
#   Example call:
#    python train_model.py

# Imports
import numpy as np
import pandas as pd


def get_data(path):
    """Load csv file.
    
    Function to load csv file with the 1994 U.S. Census data using Pandas. 
    -----------
    Parameters:
     path - Full path to the csv file with the dataset
    Returns:
     data - Full dataframe
     prices - Column with datapoints to be the targets
     features - Dataframe without the target column to be the features.
    """

    # Load the data with Pandas
    data = pd.read_csv(path)
    # Call function to load data

    # Display data
    print("*** Load data for training ***\nView first 5 rows:")
    print(data.head())

    # Compute some percentage and print them
    n_greater_50k = len(data.groupby('income').get_group('>50K'))
    n_at_most_50k = len(data.groupby('income').get_group('<=50K'))
    greater_percent = round((float(n_greater_50k) / float(len(data)) * 100), 2)

    print("*** Explore data ***")
    print("Total number of datapoints: {}".format(len(data)))
    print("Individuals making more than $50,000: {}".format(n_greater_50k))
    print("Individuals making at most $50,000: {}".format(n_at_most_50k))
    print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))

    return data


# Call function to load data
data = get_data('data/census.csv')
