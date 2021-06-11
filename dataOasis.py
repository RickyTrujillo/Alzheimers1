import json
import csv
import numpy as np
import glob
import nibabel as nib
# import matplotlib.pyplot as plt

def Clinical_Data(directory):
    temp1 = []
    Patient_Info = []
    with open('//Users//rickytrujillo/Desktop/School Files/Research/Alzheimers/ClinicalData.csv') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            temp1.clear()
            json.dumps(temp1.append(row['ADRC_ADRCCLINICALDATA ID']))
            for f1 in temp1:
                Patient_Info.append([f1[f1.find("_d")+2:], row['Subject'][int(row['Subject'].find("OAS"))+3:], row['mmse']])
    return Patient_Info

def MRI_Scans(directory):
    # go through directory and open folders containing anat1 NIFTI scans and append images, ID, and date labels to list mri_slices[]
    filenames = glob.glob(directory)
    mri_slices = []
    for files in filenames:
        data = nib.load(files)
        img = data.get_fdata()
        if np.shape(img) == (256, 256, 36):
            A = data.header['aux_file'].tostring().decode("utf-8")
            if A.find('OAS') != -1:
                B = A[int(A.find("OAS") + 3):int(A.find("_M"))]
                C = A[int(A.find("_d") + 2):]
                mri_slices.append([img[:, :, 18], B, C.rstrip('\x00')])
                # plt.imshow(data.get_fdata()[:, :, 18])
                # plt.show(block=True)
    return mri_slices
