'''
Created by Paul Coen
Usage:
    Use by calling script name or (using script package in atom) [Ctrl]+[Shift]+[B]

Description:
    This script is intended to clean the data from the following datasets:
        Chicago Crime Statistics as crime.csv
        Chicago Food Inspections as food_inspections.csv
'''

import csv
import os

script_dir = os.path.dirname(__file__)

# Get paths to each csv file
crime_path = os.path.join(script_dir, '../datasets/crime.csv')
ordered_crime_path = os.path.join(script_dir, '../datasets/ordered_crime.csv')
food_inspections_path = os.path.join(script_dir, '../datasets/food_inspections.csv')
ordered_food_inspections_path = os.path.join(script_dir, '../datasets/ordered_food_inspections.csv')

'''
# Process the crime.csv and create ordered_crime.csv
# Resulting columns are:
# [0] = date
# [1] = block
# [2] = iucr
# [3] = arrest
# [4] = domestic
# [5] = latitude
# [6] = longitude
'''
with open(crime_path) as crime_csv_file:
    with open(ordered_crime_path, mode='w') as ordered_crime_csv_file:
        crime_csv_reader = csv.reader(crime_csv_file, delimiter=',')
        ordered_crime_writer = csv.writer(ordered_crime_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        line_count = 0
        lines_removed = 0
        for row in crime_csv_reader:
            if line_count == 0:
                ordered_crime_writer.writerow(["date", "block", "iucr", "arrest", "domestic", "latitude", "longitude"])
                line_count += 1
            else:
                if (row[19] in (None, "")) or (row[20] in (None, "")):
                    print(f'Blank Data found! Removing line_count={line_count}')
                    lines_removed += 1
                elif (float(row[19]) > 42.5) or (float(row[19]) < 41.2) or (float(row[20]) < -88.4) or (float(row[20]) > -87.1):
                    print(f'Lat-Long point outside of main city! Removing line_count={line_count}')
                    lines_removed += 1
                else:
                    ordered_crime_writer.writerow([row[2], row[3], int(row[4][:2]), row[8], row[9], row[19], row[20]])
                line_count += 1
        print(f'Processed {line_count} lines. {lines_removed} lines removed.')

'''
# Process the food_inspections.csv and create ordered_food_inspections.csv
# Resulting columns are:
# [0] = dba_name
# [1] = aka_name
# [2] = facility_type
# [3] = risk
# [4] = inspection_date
# [5] = inspection_type
# [6] = results
# [7] = violations
# [8] = latitude
# [9] = longitude
'''
'''
with open(food_inspections_path) as food_inspections_csv_file:
    with open(ordered_food_inspections_path, mode='w') as ordered_food_inspections_csv_file:
        food_inspections_csv_reader = csv.reader(food_inspections_csv_file, delimiter=',')
        ordered_food_inspections_writer = csv.writer(ordered_food_inspections_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        line_count = 0
        for row in food_inspections_csv_reader:
            if line_count == 0:
                ordered_food_inspections_writer.writerow(["dba_name","aka_name","facility_type","risk","inspection_date","inspection_type","results","violations","latitude","longitude"])
                line_count += 1
            else:
                ordered_food_inspections_writer.writerow([row[1], row[2], row[4], row[5], row[10], row[11], row[12], row[13], row[14], row[15]])
                line_count += 1
        print(f'Processed {line_count} lines.')
'''
