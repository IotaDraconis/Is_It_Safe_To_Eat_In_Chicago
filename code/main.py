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

crimeAll_array = np.genfromtxt(ordered_crime_path, delimiter=',', dtype=None, encoding='utf8')
print(f'array was created, making dataframe')
crimeAll_df = pd.DataFrame(data=crimeAll_array, columns=["date", "block", "iucr", "arrest", "domestic", "latitude", "longitude"])
crimeAll_df.reset_index()
print(f'dataframe was created, taking sample')

crime01_df = crimeAll_df.loc[crimeAll_df['iucr'] == '1']
crime03_df = crimeAll_df.loc[crimeAll_df['iucr'] == '3']
crime04_df = crimeAll_df.loc[crimeAll_df['iucr'] == '4']
crime08_df = crimeAll_df.loc[crimeAll_df['iucr'] == '8']

crime_sample1 = crime01_df.sample(n=5000, replace=True)
crime_sample2 = crime03_df.sample(n=5000, replace=True)
crime_sample3 = crime04_df.sample(n=5000, replace=True)
crime_sample4 = crime08_df.sample(n=5000, replace=True)

# Plot the samples
# Create the subplots for each sample
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
print(f'After Subplot')
axs[0, 0].plot(crime_sample1.longitude, crime_sample1.latitude, 'r,')
print(f'After Plot 1')
axs[0, 1].plot(crime_sample2.longitude, crime_sample2.latitude, 'b,')
print(f'After Plot 2')
axs[1, 0].plot(crime_sample3.longitude, crime_sample3.latitude, 'g,')
print(f'After Plot 3')
axs[1, 1].plot(crime_sample4.longitude, crime_sample4.latitude, 'm,')
print(f'After Plot 4')
plt.show()
