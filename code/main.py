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

crimeData_array = np.genfromtxt(ordered_crime_path, delimiter=',', dtype=None, encoding='utf8')
print(f'array was created, making dataframe')
crimeData_dataframe = pd.DataFrame(data=crimeData_array, columns=["date", "block", "iucr", "arrest", "domestic", "latitude", "longitude"])
crimeData_dataframe.reset_index()
print(f'dataframe was created, taking sample')

crimeData_sample1 = crimeData_dataframe.sample(n=75000, replace=True)
crimeData_sample2 = crimeData_dataframe.sample(n=75000, replace=True)
crimeData_sample3 = crimeData_dataframe.sample(n=75000, replace=True)
crimeData_sample4 = crimeData_dataframe.sample(n=75000, replace=True)

# Plot the samples
# Create the subplots for each sample
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

axs[0, 0].plot(crimeData_sample1.longitude, crimeData_sample1.latitude, 'b,')
axs[0, 1].plot(crimeData_sample2.longitude, crimeData_sample2.latitude, 'r,')
axs[1, 0].plot(crimeData_sample3.longitude, crimeData_sample3.latitude, 'g,')
axs[1, 1].plot(crimeData_sample4.longitude, crimeData_sample4.latitude, 'm,')
plt.show()
