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
from sklearn.cluster import DBSCAN
from sklearn import metrics

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

#Note: The 01 set is <10K items, thus it does not need samples to express it.
crime_sample1 = crime01_df
crime_sample2 = crime03_df.sample(n=75000, replace=True)
crime_sample3 = crime04_df.sample(n=75000, replace=True)
crime_sample4 = crime08_df.sample(n=75000, replace=True)

# create arrays with just the data we ned for DBSCAN
X = crime_sample1[['latitude', 'longitude']]

# Run DBSCAN on the samples
# Need to tweak eps
db = DBSCAN(eps=0.0065, min_samples=50).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)


# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    print(xy.latitude)
    plt.plot(xy.latitude,
             xy.longitude,
             '.',
             markerfacecolor=tuple(col),
             markeredgecolor='k',
             markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy.latitude,
             xy.longitude,
             '.',
             markerfacecolor=tuple(col),
             markeredgecolor='k',
             markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()



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
