'''
Created by Paul Coen
This script is to be used to run the clutering algorithms for CS-533 on the Chicago datasets we are using
If not already done, please download the following datasets described in datasets/DatasetList.txt
Then run prepare.py. This will clean the data for use with this script
'''

#import csv
import os
import scipy.cluster as cluster
import matplotlib.pyplot as plt
import numpy as np

# Read entire csv as a numpy array
ordered_crime_path = os.path.join(os.path.dirname(__file__), '../datasets/ordered_crime.csv')
#ordered_food_inspections_path = os.path.join(os.path.dirname(__file__), '../datasets/ordered_food_inspections.csv')

crimeData = np.genfromtxt(ordered_crime_path, delimiter=',', dtype=None, encoding='utf8')
#plt.plot(crimeData[1:100, 5], crimeData[1:100, 6], 'ro')
#plt.axis([0, 180, 0, 180])
#plt.show()
