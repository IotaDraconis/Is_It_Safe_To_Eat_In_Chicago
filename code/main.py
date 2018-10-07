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
import pandas as pd

# Read entire csv as a numpy array
ordered_crime_path = os.path.join(os.path.dirname(__file__), '../datasets/ordered_crime.csv')
#ordered_food_inspections_path = os.path.join(os.path.dirname(__file__), '../datasets/ordered_food_inspections.csv')

#crimeData_array = np.genfromtxt(ordered_crime_path, delimiter=',', dtype=None, encoding='utf8')
#print(f'array was created, making dataframe')
#crimeData_dataframe = pd.dataframe(data=crimeData_array, columns=["date", "block", "iucr", "arrest", "domestic", "latitude", "longitude"])
#print(f'dataframe was created, taking sample')


#crimeData_sample = crimeData_dataframe.sample(n=1000, replace=True)
#print(f'crimedata_sample[0] = {crimeData_sample[1]}')

# Plot the sample
#plt.plot(crimeData_array[1:100, 5], crimeData_array[1:100, 6], 'ro')
#plt.show()
