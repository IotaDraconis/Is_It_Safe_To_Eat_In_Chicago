'''
Created by Paul Coen
This script is to be used to run the clutering algorithms for CS-533 on the Chicago datasets we are using
If not already done, please download the following datasets described in datasets/DatasetList.txt
Then run prepare.py. This will clean the data for use with this script
'''

#import csv
import os
import matplotlib
import numpy as np
import pandas as pd
import scipy.cluster as cluster
matplotlib.use('Qt5Agg') # Note, the backend of matplotlib should be Qt5Agg for it to work with 75K points
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import DBSCAN
from sklearn import metrics

## Function Defs
def find_data_DBSCAN(plt, sample, epsln, minSam, title):
    X = sample[['latitude', 'longitude']]
    # Run DBSCAN on the samples
    # Need to tweak eps
    db = DBSCAN(eps=epsln, min_samples=minSam).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(f'Estimated number of clusters for {title}: {n_clusters_}' )

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy.latitude,
                 xy.longitude,
                 ',',
                 markerfacecolor=tuple(col),
                 markeredgecolor='k',
                 markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy.latitude,
                 xy.longitude,
                 ',',
                 markerfacecolor=tuple(col),
                 markeredgecolor='k',
                 markersize=6)
    plt.set_title(f'{title}:{n_clusters_}')


#TODO: Implement sow_and_grow with the disjoint-set data structure
def sow_and_grow(X, eps, min_samples, n):
    pass

def mesh_plot():
    pass

## Start of lodaing data
# Read entire csv as a numpy array
ordered_crime_path = os.path.join(os.path.dirname(__file__), '../datasets/ordered_crime.csv')
#ordered_food_inspections_path = os.path.join(os.path.dirname(__file__), '../datasets/ordered_food_inspections.csv')

crimeAll_array = np.genfromtxt(ordered_crime_path, delimiter=',', dtype=None, encoding='utf8')
print(f'array was created, making dataframe')
crimeAll_df = pd.DataFrame(data=crimeAll_array, columns=["date", "block", "iucr", "arrest", "domestic", "latitude", "longitude"])
crimeAll_df.reset_index()
print(f'dataframe was created, taking sample')

crime_sample = {}
for i in range(16):
    print(f'Using ICUR: {i + 1}')
    crime_sample_TEMP = crimeAll_df.loc[crimeAll_df['iucr'] == f'{i + 1}']
    print(type(crime_sample_TEMP))
    if crime_sample_TEMP.shape[0] > 50000:
        print(f'{i} has more than 50K points({crime_sample_TEMP.shape[0]} points), using a sample of 50K points.')
        crime_sample[i] = crime_sample_TEMP.sample(n=50000, replace=True)
    elif crime_sample_TEMP.shape[0] > 0:
        print(f'{i} has {crime_sample_TEMP.shape[0]} points, using all points.')
        crime_sample[i] = crime_sample_TEMP
    elif crime_sample_TEMP.shape[0] <= 0:
        #crime_sample[{i}] = crime_sample_TEMP
        print(f'{i} is invalid, no points were found')

## Run the sets created above on DBSCAN
fig, axs = plt.subplots(4, 4, sharex=True, sharey=True)
# The values for eps and min_samples need tweaked
find_data_DBSCAN(axs[0, 0], crime_sample[0], 0.0065, 50, '01 Homicide')

#These all need their parameters tweaked better
find_data_DBSCAN(axs[1, 0], crime_sample[1],  0.0065, 50, 'Criminal Sexual Assault')
find_data_DBSCAN(axs[2, 0], crime_sample[2],  0.0065, 50, 'Robbery')
find_data_DBSCAN(axs[3, 0], crime_sample[3],  0.0065, 50, 'Battery')
find_data_DBSCAN(axs[0, 1], crime_sample[4],  0.0065, 50, 'Assault')
find_data_DBSCAN(axs[1, 1], crime_sample[5],  0.0065, 50, 'Burglary')
#find_data_DBSCAN(axs[2, 1], crime_sample[6],  0.0065, 50, 'Vehicular Burglary')
find_data_DBSCAN(axs[3, 1], crime_sample[7],  0.0065, 50, 'Theft')
find_data_DBSCAN(axs[0, 2], crime_sample[8],  0.0065, 50, 'Motor Vehicle Theft')
find_data_DBSCAN(axs[1, 2], crime_sample[9],  0.0065, 50, 'Arson & Human Trafficking')
find_data_DBSCAN(axs[2, 2], crime_sample[10], 0.0065, 50, 'Deceptive Practices')
find_data_DBSCAN(axs[3, 2], crime_sample[11], 0.0065, 50, 'Computer Deceptive Practices')
find_data_DBSCAN(axs[0, 3], crime_sample[12], 0.0065, 50, 'Criminal Damage & Tresspassing')
find_data_DBSCAN(axs[1, 3], crime_sample[13], 0.0065, 50, 'Deadly Weapons')
find_data_DBSCAN(axs[2, 3], crime_sample[14], 0.0065, 50, 'Sex Offenses')
find_data_DBSCAN(axs[3, 3], crime_sample[15], 0.0065, 50, 'Gambling')

plt.show()
print(f'Done plotting');

'''
# Plot the samples
# Create the subplots for each sample
# Uncomment the bellow code if you would rather have a dark background for the plot
#plt.style.use(['dark_background'])
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
'''
