import csv
import math
import pandas as pd
from pandas import DataFrame
from operator import itemgetter
import itertools


def combine_lists(mri_slices, Patient_Info, first_csv_dir):
    # Interpolating missing mmse slices with various cases (linear interpolation t-1:t+1, t-1:t+2, manual inputs, assumptions
    with open(first_csv_dir, 'w', newline='') as new_csvfile:
        csv_combined = csv.writer(new_csvfile)
        csv_combined.writerow(['ID', 'Date', 'mmse'])
        # Patient Info [date, ID, mmse] & mri_slices [img, ID, date]
        # check to see if ID match and then input into csv file as [ID date mmse]
        for i in range(len(Patient_Info)):
            unwritten = 0
            PatID = Patient_Info[i][1]
            for j in range(len(mri_slices)):
                if Patient_Info[i][1] == mri_slices[j][1]:
                    if unwritten == 0:
                        # print([Patient_Info[i][1], Patient_Info[i][0], Patient_Info[i][2]])
                        csv_combined.writerow([Patient_Info[i][1], Patient_Info[i][0], Patient_Info[i][2]])
                        unwritten = 1
                    if PatID != Patient_Info[i - 1][1] or i == 0:
                        csv_combined.writerow([mri_slices[j][1], mri_slices[j][2], ''])


# def interpolate_csv(first_csv_dir, interpolated_csv_dir):
#     pandas_reader = pd.read_csv(first_csv_dir)
#     csv_sorted = pandas_reader.sort_values(["ID", "Date"]).reset_index(drop=True)
#
#     for i in range(1, len(csv_sorted)):
#         if math.isnan(csv_sorted.loc[i]['mmse']):
#             if csv_sorted.loc[i + 1]['ID'] != csv_sorted.loc[i]['ID']:
#                 csv_sorted.loc[i, 'mmse'] = csv_sorted.iloc[i - 1][2]
#             elif math.isnan(csv_sorted.loc[i + 1]['mmse']) and csv_sorted.loc[i + 1]['ID'] != csv_sorted.loc[i + 2][
#                 'ID']:
#                 csv_sorted.loc[i, 'mmse'] = csv_sorted.iloc[i - 1][2]
#                 csv_sorted.loc[i + 1, 'mmse'] = csv_sorted.iloc[i - 2][2]
#             elif math.isnan(csv_sorted.loc[i + 1]['mmse']):
#                 csv_sorted.loc[i, 'mmse'] = ((csv_sorted.iloc[i][1] - csv_sorted.iloc[i - 1][1]) * (
#                         csv_sorted.iloc[i + 2][2] - csv_sorted.iloc[i - 1][2])) / (
#                                                 (csv_sorted.iloc[i + 2][1] - csv_sorted.iloc[i - 1][1])) + \
#                                             csv_sorted.iloc[i - 1][2]
#             elif csv_sorted.loc[i + 1]['ID'] == csv_sorted.loc[i]['ID']:
#                 csv_sorted.loc[i, 'mmse'] = ((csv_sorted.iloc[i][1] - csv_sorted.iloc[i - 1][1]) * (
#                         csv_sorted.iloc[i + 1][2] - csv_sorted.iloc[i - 1][2])) / (
#                                                 (csv_sorted.iloc[i + 1][1] - csv_sorted.iloc[i - 1][1])) + \
#                                             csv_sorted.iloc[i - 1][2]

def interpolate_csv(first_csv_dir, interpolated_csv_dir):
    pandas_reader = pd.read_csv(first_csv_dir)
    csv_sorted = pandas_reader.sort_values(["ID", "Date"]).reset_index(drop=True)

    for i in range(1, len(csv_sorted)):
        if math.isnan(csv_sorted.loc[i]['mmse']):
            if math.isnan(csv_sorted.loc[i+1]['mmse']) == False and csv_sorted.loc[i+1]['ID'] == csv_sorted.loc[i]['ID'] == csv_sorted.loc[i-1]['ID']:
                csv_sorted.loc[i, 'mmse'] = ((csv_sorted.iloc[i][1] - csv_sorted.iloc[i - 1][1]) * (csv_sorted.iloc[i + 1][2] - csv_sorted.iloc[i - 1][2])) / (
                                (csv_sorted.iloc[i + 1][1] - csv_sorted.iloc[i - 1][1])) + \
                                                     csv_sorted.iloc[i - 1][2]

            if math.isnan(csv_sorted.loc[i + 1]['mmse']) and csv_sorted.loc[i+2]['ID'] ==csv_sorted.loc[i]['ID'] and csv_sorted.loc[i+2]['ID'] == csv_sorted.loc[i]['ID'] == csv_sorted.loc[i-1]['ID']:
                csv_sorted.loc[i, 'mmse'] = ((csv_sorted.iloc[i][1] - csv_sorted.iloc[i - 1][1]) * (
                                csv_sorted.iloc[i + 2][2] - csv_sorted.iloc[i - 1][2])) / (
                                                         (csv_sorted.iloc[i + 2][1] - csv_sorted.iloc[i - 1][1])) + \
                                                     csv_sorted.iloc[i - 1][2]

            if math.isnan(csv_sorted.loc[i + 2]['mmse']) and csv_sorted.loc[i+2]['ID'] == csv_sorted.loc[i]['ID']:
                csv_sorted.loc[i, 'mmse'] = csv_sorted.loc[i-1, 'mmse']

            if csv_sorted.loc[i + 1]['ID'] != csv_sorted.loc[i]['ID']:
                csv_sorted.loc[i, 'mmse'] = csv_sorted.iloc[i - 1][2]

            if math.isnan(csv_sorted.loc[i + 1]['mmse']) and csv_sorted.loc[i + 1]['ID'] != csv_sorted.loc[i + 2]['ID']:
                csv_sorted.loc[i, 'mmse'] = csv_sorted.iloc[i - 1][2]
                csv_sorted.loc[i + 1, 'mmse'] = csv_sorted.iloc[i - 2][2]

        if math.isnan(csv_sorted.loc[i]['mmse']) and csv_sorted.loc[i]['Date'] == 0:
            if csv_sorted.loc[i]['ID'] == 30194:
                csv_sorted.iloc[i:i + 7, 2] = 30.0
            elif csv_sorted.loc[i]['ID'] == 30314:
                csv_sorted.loc[i:i + 7, 'mmse'] = 30.0
            elif csv_sorted.loc[i]['ID'] == 30375:
                csv_sorted.loc[i, 'mmse'] = 28.0
            elif csv_sorted.loc[i]['ID'] == 30505:
                csv_sorted.loc[i, 'mmse'] = 28.0
            elif csv_sorted.loc[i]['ID'] == 30675:
                csv_sorted.loc[i:i + 10, 'mmse'] = 29.0
            elif csv_sorted.loc[i]['ID'] == 30718:
                csv_sorted.loc[i:i + 4, 'mmse'] = 28.0
            elif csv_sorted.loc[i]['ID'] == 30805:
                csv_sorted.loc[i:i + 2, 'mmse'] = 30.0
            elif csv_sorted.loc[i]['ID'] == 30825:
                csv_sorted.loc[i:i + 1, 'mmse'] = 27.0
            elif csv_sorted.loc[i]['ID'] == 30811:
                csv_sorted.loc[i, 'mmse'] = 28.0
            elif csv_sorted.loc[i]['ID'] == 30829:
                csv_sorted.loc[i:i + 7, 'mmse'] = 30.0
            elif csv_sorted.loc[i]['ID'] == 30923:
                csv_sorted.loc[i:i + 4, 'mmse'] = 29.0
            elif csv_sorted.loc[i]['ID'] == 30936:
                csv_sorted.loc[i:i + 6, 'mmse'] = 29.0
            elif csv_sorted.loc[i]['ID'] == 31002:
                csv_sorted.loc[i, 'mmse'] = 27.0
            elif csv_sorted.loc[i]['ID'] == 31035:
                csv_sorted.loc[i:i + 1, 'mmse'] = 27.0
            elif csv_sorted.loc[i]['ID'] == 31108:
                csv_sorted.loc[i, 'mmse'] = 30.0
            elif csv_sorted.loc[i]['ID'] == 31155:
                csv_sorted.loc[i:i + 8, 'mmse'] = 30.0
            elif csv_sorted.loc[i]['ID'] == 31160:
                csv_sorted.loc[i, 'mmse'] = 29.0
    csv_sorted.loc[5662:5663, 'mmse'] = 29.0
    csv_sorted.loc[2510:2512, 'mmse'] = 29.0
    # print(csv_sorted)
    csv_sorted.reset_index(drop=True).to_csv(interpolated_csv_dir)
    test2 = pd.read_csv(interpolated_csv_dir).values.tolist()

    for j in range(len(test2)):
        if math.isnan(test2[j][3]):
            print(j)

def mri_to_mmse(mri_slices, interpolated_csv_dir):
    # # pair up mri_slices inputs to csv data matches with corresponding image and mmse score
    inter_csv_read = pd.read_csv(interpolated_csv_dir)
    inter_csv = inter_csv_read.values.tolist()

    temp = []
    data = []
    for i in range(len(mri_slices)):
        for j in range(len(inter_csv)):
            if int(mri_slices[i][1]) == int(inter_csv[j][1]) and int(mri_slices[i][2]) == int(inter_csv[j][2]):
                temp.append([mri_slices[i][0], mri_slices[i][1], mri_slices[i][2], inter_csv[j][1], inter_csv[j][2], int(inter_csv[j][3])])

    data_sorted = sorted(temp, key=itemgetter(1, 2))
    for i in range(len(data_sorted)):
        unwritten = 0
        if i == 0:
            data.append(data_sorted[i])
            data.pop(0)
        if data_sorted[i][1] == data_sorted[i - 1][1] and data_sorted[i][2] == data_sorted[i - 1][2]:
            pass
        else:
            if unwritten == 0:
                data.append(data_sorted[i])
                unwritten = 1

    # df = DataFrame(data, columns=['Image', 'ID', 'Date', 'ID2', 'Date2', 'mmse'])
    # sorted_df = df.sort_values(['ID', 'Date'])
    # paired_csv = sorted_df.drop_duplicates(subset=['ID', 'Date'], keep='last', inplace=False)
    # # paired_csv.to_csv('//Users//rickytrujillo/Desktop//NewCombinedData_VIEW1.csv')
    return data

#def convert_to_factor(data):
