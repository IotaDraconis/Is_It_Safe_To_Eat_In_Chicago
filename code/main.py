'''
Created by Paul Coen
This script is to be used to run the clutering algorithms for CS-533 on the Chicago datasets we are using
If not already done, please download the following datasets described in datasets/DatasetList.txt
Then run prepare.py. This will clean the data for use with this script
'''

# DEBUG import csv
import os

import matplotlib
matplotlib.use('Qt5Agg')  # Note, the backend of matplotlib should be Qt5Agg for it to work with 75K points
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster as cluster
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from sklearn import metrics
from sklearn.cluster import DBSCAN


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
    print(f'Estimated number of clusters for {title}: {n_clusters_}')

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
    plt.set_title(f'{title}:{n_clusters_}')


def mesh_plot(plt, X, Y, Z, title):
    plt.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.7, cmap=cm.jet)
    # DEBUG plt.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    plt.contourf(X, Y, Z, zdir='z', cmap=cm.jet, offset=-400)  # These used to use coolwarm
    plt.contourf(X, Y, Z, zdir='x', cmap=cm.jet, offset=41.6)
    plt.contourf(X, Y, Z, zdir='y', cmap=cm.jet, offset=-87.4)
    plt.set_xlabel('Lat')
    plt.set_xlim(41.6, 42.2)
    plt.set_ylabel('Long')
    plt.set_ylim(-88.0, -87.4)
    plt.set_zlabel('Number of Crimes')
    plt.set_zlim(-100, 1000)
    plt.set_title(f'{title}')

print("test")
# Food Data
ordered_food_inspections_path = os.path.join(os.path.dirname(__file__), '../datasets/ordered_food_inspections.csv')
food_array = np.genfromtxt(ordered_food_inspections_path, delimiter=',', dtype=None, encoding='utf8')
food_df = pd.DataFrame(data=food_array, columns=["risk", "latitude", "longitude"])
food_df.reset_index()

food_sample = {}
for i in range(3):
    last_string_section = ""
    if i == 0:
        last_string_section = " (High)"
    elif i == 1:
        last_string_section = " (Medium)"
    elif i == 2:
        last_string_section = " (Low)"
    risk_string = "Risk " + f'{(i + 1)}' + last_string_section
    print(risk_string)
    food_df_TEMP = food_df.loc[food_df['risk'] == f'{risk_string}']
    if food_df_TEMP.shape[0] > 50000:
        print(f'{i} has more than 50K points({food_df_TEMP.shape[0]} points), using a sample of 50K points.')
        food_sample[i] = food_df_TEMP.sample(n=50000, replace=True)
        food_sample[i].index = range(len(food_sample[i].index))
    elif food_df_TEMP.shape[0] > 0:
        print(f'{i} has {food_df_TEMP.shape[0]} points, using all points.')
        food_sample[i] = food_df_TEMP
        food_sample[i].index = range(len(food_sample[i].index))
    elif food_df_TEMP.shape[0] <= 0:
        print(f'{i} is invalid, no points were found')

'''
fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
# The values for eps and min_samples need tweaked
find_data_DBSCAN(axs[0], food_sample[0], 0.0065, 50, 'Risk 1 (High)')
find_data_DBSCAN(axs[1], food_sample[1], 0.0065, 50, 'Risk 2 (Medium)')
find_data_DBSCAN(axs[2], food_sample[2], 0.0065, 50, 'Risk 3 (Low)')

plt.show()
'''

max_lat = 42.2
min_lat = 41.6

max_long = -87.4
min_long = -88.0

sep_distance = 0.6  # was 1.3, this was reduced due to the changing of points

# DON'T SET separation_value TOO LOW!!! It controls the size of the grid directly
# Range should be somewhere between 0.13 (11 * 11 grids) and 0.0013 (1001 * 1001 grids)
separation_value = 0.006  # 101 * 101 grids
grid_value = round(sep_distance / separation_value) + 1
food_meshX = np.ndarray(shape=(grid_value, grid_value), dtype=float)
food_meshY = np.ndarray(shape=(grid_value, grid_value), dtype=float)
food_meshZ = np.ndarray(shape=(16, grid_value, grid_value), dtype=float)
# Create the meshes

for x in range(grid_value):
    for y in range(grid_value):
        food_meshX[x][y] = min_lat + (y * separation_value)
        food_meshY[y][x] = max_long + (-1 * y * separation_value)

for i in range(16):
    if i != 6:
        for x in range(grid_value):
            for y in range(grid_value):
                food_meshZ[i][x][y] = 0

# Create crime arrays adding needed heights
for i in range(3):
    for j in range(food_sample[i].shape[0]):
        # DEBUG print(food_sample[i].iloc[j])
        # DEBUG print(f'Lat: [{round((42.5 - float(food_sample[i].iloc[j].latitude))/1.3 * 100)}] Long: [{round(-1 * (-88.4 - float(food_sample[i].iloc[j].longitude))/1.3 * 100)}]')
        food_meshZ[i][round((max_lat - float(food_sample[i].iloc[j].latitude)) / sep_distance * (grid_value - 1))][round(-1 * (min_long - float(food_sample[i].iloc[j].longitude)) / sep_distance * (grid_value - 1))] += 1

plot_titles = ["Risk 1 (High)",
               "Risk 2 (Medium)",
               "Risk 3 (Low)"]

# Run the sets through our mesh plotting
all_in_one_plot = False


print(f'plotting meshes')
if all_in_one_plot is True:
    fig = plt.figure(num=i, figsize=plt.figaspect(0.5))


for i in range(3):
    if all_in_one_plot is True:
        axs = fig.add_subplot(1, 3, i + 1, projection='3d')
    elif all_in_one_plot is False:
        fig = plt.figure(num=i, figsize=plt.figaspect(0.5))
        axs = fig.add_subplot(1, 1, 1, projection='3d')  # used to use 4, 4, i + 1 so that they were all on the same plot as subplots

    print(f'Running mesh function {plot_titles[i]} with x:{food_meshX.shape} y:{food_meshY.shape} z{food_meshZ.shape}')
    mesh_plot(axs, food_meshX, food_meshY, food_meshZ[i], plot_titles[i])
    print(f'Done plotting {plot_titles[i]}')

plt.show()
print(f'Done plotting')


print("Remove the bellow line and re-run to run the crime set")
# This will be re-organized into two scripts over break
pause_processing = input("PRESS ENTER TO CONTINUE.")


# Crime Data
# Read entire csv as a numpy array
ordered_crime_path = os.path.join(os.path.dirname(__file__), '../datasets/ordered_crime.csv')

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
        crime_sample[i].index = range(len(crime_sample[i].index))
    elif crime_sample_TEMP.shape[0] > 0:
        print(f'{i} has {crime_sample_TEMP.shape[0]} points, using all points.')
        crime_sample[i] = crime_sample_TEMP
        crime_sample[i].index = range(len(crime_sample[i].index))
    elif crime_sample_TEMP.shape[0] <= 0:
        print(f'{i} is invalid, no points were found')

plot_titles = ['Homicide',
               'Criminal Sexual Assault',
               'Robbery',
               'Battery',
               'Assault',
               'Burglary',
               'None',
               'Theft',
               'Motor Vehicle Theft',
               'Arson & Human Trafficking',
               'Deceptive Practices',
               'Computer Deceptive Practices',
               'Criminal Damage & Tresspassing',
               'Deadly Weapons',
               'Sex Offenses',
               'Gambling']

'''
## Run the sets created above on DBSCAN
fig, axs = plt.subplots(4, 4, sharex=True, sharey=True)
# The values for eps and min_samples need tweaked
find_data_DBSCAN(axs[0, 0], crime_sample[0], 0.0065, 50, 'Homicide')

#These all need their parameters tweaked better
find_data_DBSCAN(axs[1, 0], crime_sample[1],  0.0065, 50, 'Criminal Sexual Assault')
find_data_DBSCAN(axs[2, 0], crime_sample[2],  0.0065, 50, 'Robbery')
find_data_DBSCAN(axs[3, 0], crime_sample[3],  0.0065, 50, 'Battery')
find_data_DBSCAN(axs[0, 1], crime_sample[4],  0.0065, 50, 'Assault')
find_data_DBSCAN(axs[1, 1], crime_sample[5],  0.0065, 50, 'Burglary')
# Skip plot 7 (index 6) due to it having no data
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

# Initial plot of city without height:
# DEBUG upper_left_point = (-88.4, 42.5)
# DEBUG lower_right_point = (-87.1, 41.2)

# Majority of data actually falls within this range:
#  41.6,  42.2
# -88.0, -87.4

max_lat = 42.2
min_lat = 41.6

max_long = -87.4
min_long = -88.0

sep_distance = 0.6  # was 1.3, this was reduced due to the changing of points

# DON'T SET separation_value TOO LOW!!! It controls the size of the grid directly
# Range should be somewhere between 0.13 (11 * 11 grids) and 0.0013 (1001 * 1001 grids)
separation_value = 0.006  # 101 * 101 grids
grid_value = round(sep_distance / separation_value) + 1
crime_meshX = np.ndarray(shape=(grid_value, grid_value), dtype=float)
crime_meshY = np.ndarray(shape=(grid_value, grid_value), dtype=float)
crime_meshZ = np.ndarray(shape=(16, grid_value, grid_value), dtype=float)
# Create the meshes

for x in range(grid_value):
    for y in range(grid_value):
        crime_meshX[x][y] = min_lat + (y * separation_value)
        crime_meshY[y][x] = max_long + (-1 * y * separation_value)

for i in range(16):
    if i != 6:
        for x in range(grid_value):
            for y in range(grid_value):
                crime_meshZ[i][x][y] = 0

# Create crime arrays adding needed heights
for i in range(16):
    if i != 6:
        for j in range(crime_sample[i].shape[0]):
            # DEBUG print(crime_sample[i].iloc[j])
            # DEBUG print(f'Lat: [{round((42.5 - float(crime_sample[i].iloc[j].latitude))/1.3 * 100)}] Long: [{round(-1 * (-88.4 - float(crime_sample[i].iloc[j].longitude))/1.3 * 100)}]')
            crime_meshZ[i][round((max_lat - float(crime_sample[i].iloc[j].latitude)) / sep_distance * (grid_value - 1))][round(-1 * (min_long - float(crime_sample[i].iloc[j].longitude)) / sep_distance * (grid_value - 1))] += 1
    else:
        print(f'Skipping {plot_titles[6]}')


# Run the sets through our mesh plotting
all_in_one_plot = False


print(f'plotting meshes')
if all_in_one_plot is True:
    fig = plt.figure(num=i, figsize=plt.figaspect(0.5))


for i in range(16):
    if all_in_one_plot is True:
        axs = fig.add_subplot(4, 4, i + 1, projection='3d')
    elif all_in_one_plot is False:
        fig = plt.figure(num=i, figsize=plt.figaspect(0.5))
        axs = fig.add_subplot(1, 1, 1, projection='3d')  # used to use 4, 4, i + 1 so that they were all on the same plot as subplots

    if i != 6:
        print(f'Running mesh function {plot_titles[i]} with x:{crime_meshX.shape} y:{crime_meshY.shape} z{crime_meshZ.shape}')
        mesh_plot(axs, crime_meshX, crime_meshY, crime_meshZ[i], plot_titles[i])
        print(f'Done plotting {plot_titles[i]}')
    else:
        print(f'Skipping {plot_titles[6]}')

plt.show()
print(f'Done plotting')

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
