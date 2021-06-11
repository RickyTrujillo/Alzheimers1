import dataOasis
import csvOasis
import data_preprocessing
import Googlenet
import pandas as pd
import numpy as np


def main():
    clinical_dir = '//Users//rickytrujillo/Desktop/School Files/Research/Alzheimers/ClinicalData.csv'
    #mri_dir = '/Users/rickytrujillo/Desktop/School Files/Research/Alzheimers/central.xnat.org/*/anat1/NIFTI/*nii.gz'
    mri_dir = '/Users/rickytrujillo/Desktop/School Files/Research/Alzheimers/OASIScopy/*/anat1/NIFTI/*nii.gz'
    first_csv_dir = '//Users//rickytrujillo/Desktop/School Files/Research/Alzheimers/CombinedData1.csv'
    interpolated_dir = '//Users//rickytrujillo/Desktop/School Files/Research/Alzheimers/NewCombinedData1.csv'
    Patient_Info = dataOasis.Clinical_Data(clinical_dir)
    mri_slices = dataOasis.MRI_Scans(mri_dir)
    csvOasis.combine_lists(mri_slices,Patient_Info, first_csv_dir)
    csvOasis.interpolate_csv(first_csv_dir, interpolated_dir)
    data_sorted = csvOasis.mri_to_mmse(mri_slices, interpolated_dir)
    test_images, test_labels, valid_images, valid_labels, train_images, train_labels = data_preprocessing.data_split(data_sorted)
    Googlenet.model_architecture(test_images, test_labels, valid_images, valid_labels, train_images, train_labels)


if __name__ == "__main__":
    main()
