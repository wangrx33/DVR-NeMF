import SimpleITK as sitk
import numpy as np
import os
import cv2
import albumentations as A

# N = 30  # num of patients
# M = 20  # num of phases of each patient
# D = 20  # num of slices 
# H = 256 # height of image
# W = 256 # width of image

split = 60
beat = 4
 
def normalize(data):
    if (data.max()-data.min()) == 0:
        return data
    else:
        return ((data-data.min())/(data.max()-data.min()))

# 先把所有训练数据读到内存里
def load_mri_data(basedir):
    '''
    数据集是由N位病人，每位病人的M个分相数据，每个分相数据为一个3D Volume(D,H,W)组成，即数据总维度为(N,M,D,H,W)
    如果我们的模型是patient-specific的话，那就令N=1
    '''
    # 将全部数据读取到内存中
    # all_data = np.zeros(shape=(N,M,D,H,W))
    all_patients = []
    patientslist = os.listdir(basedir)[1:2]
    # patientslist.sort()
    for i, patient in enumerate(patientslist):
        patientpath = os.path.join(basedir, patient)
        phaselist = os.listdir(patientpath)
        phaselist.sort()
        if len(phaselist) == 1:
            all_phase = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(patientpath, phaselist[0])))
            all_phase = np.swapaxes(all_phase, 0, 3)
        elif len(phaselist) > 1:
            all_phase = []
            for j, phase in enumerate(phaselist):
                phasepath = os.path.join(patientpath, phase)
                all_phase.append(sitk.GetArrayFromImage(sitk.ReadImage(phasepath)))
            all_phase = np.stack(all_phase, 0)
        all_patients.append(all_phase)
    all_data_whole = np.stack(all_patients, 0)    # patients_num, phase_num, slice, height, width
    all_data = all_data_whole[:, :, 30:60, 30:60, 30:60]  # crop 128*128 size due to memory limitation-

    patient_num, phase_num, slice_num, height, width = all_data.shape
    # 以patient-specific为例，提取作为输入的单层slice和时刻t
    # volumes = np.zeros(shape=(phase_num, slice_num, height, width))
    # slices = np.zeros(shape=(phase_num, height, width))
    # times = np.zeros(shape=(phase_num,))

    volumes_img = np.zeros_like(all_data[0]).astype(np.float64)   # (phase_num, slice_num, height, width)
    for i in range(volumes_img.shape[0]):
        volumes_img[i] = normalize(all_data[0][i])
    volumes_k = np.zeros_like(all_data[0]).astype(np.complex128)    # (phase_num, slice_num, height, width)
    for i in range(volumes_k.shape[0]):
        volumes_k[i] = (np.fft.fftn(all_data[0][i])) / np.sqrt((1e+10)*2*np.pi)
    # slices = volumes_k[:,int(slice_num/2),:,:]  # (phase_num, height, width)
    slices_img = volumes_img[:,int(slice_num/2),:,:]    # 这里对a/s/c哪个视角作为输入没有进行分析测试
    slices_k = volumes_k[:,int(slice_num/2),:,:]
    # times = np.tile(np.arange(0, 1, beat/phase_num), beat)    # (phase_num,)
    times = np.arange(0, phase_num, 1)    # (phase_num,)

    trainset =[volumes_img[:split], volumes_k[:split], slices_img[:split], slices_k[:split], times[:split]]
    testset =[volumes_img[split:], volumes_k[split:], slices_img[split:], slices_k[split:], times[split:]]

    # return volumes_img, volumes_k, slices_img, slices_k, times
    return trainset, testset

