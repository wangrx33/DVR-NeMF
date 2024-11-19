import SimpleITK as sitk
import numpy as np
import os
import cv2

data_path = './dataset/acdc/patient002/patient002_4d.nii.gz'
data_4d = sitk.GetArrayFromImage(sitk.ReadImage(data_path))
print(data_4d.shape)    # (30, 10, 256, 216) (phase, slice, height, width)

save_path = './dataset/mri/acdc/patient002'
os.makedirs(save_path, exist_ok=True)
for i in range(data_4d.shape[0]):
    data_3d = data_4d[i]
    sitk.WriteImage(sitk.GetImageFromArray(data_3d), os.path.join(save_path, str(i).zfill(5)+'.nii.gz'))
    
