# This chunk of code selects 110 coronal slices from the baseline volume, 110 coronal slices from M06, 110 coronal slices from M12
#calculated from the middle slice of the origional volume and save a new volume of shape 110x110x330.
import numpy as np
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import glob
import nibabel as nb
import cv2
import imageio
import SimpleITK as sitk
import numpy as np
from PIL import Image
def save_images_return_paths_list(path, number_of_images, sav_img_pth):
    scan = nb.load(path)
    scan = scan.get_fdata()
    sag, axl, cor = scan.shape
    middle = cor // 2
    num_of_images_middle = number_of_images // 2
    start_from = middle - num_of_images_middle
    up_to = middle + num_of_images_middle
    single_vol = []
    for i in range(start_from, up_to):
        arr = scan[:, i, :]
        coronal_pth = os.path.join(sav_img_pth, str( i ) + '.png')
        coronal = cv2.resize(arr, (110, 110), interpolation = cv2.INTER_AREA)
        cv2.imwrite(coronal_pth, coronal)
    path = os.path.join(sav_img_pth, '*.*')
    k = []
    for i in glob.glob(path):
        ii=i.split('/')[-1].split('.')[0]
        k.append(int(ii))
    k = sorted(k)
    all_imges_pth = []
    for pth in k:
        img_pth = os.path.join(sav_img_pth, str(pth ) + '.png')
        all_imges_pth.append(img_pth)
    return all_imges_pth
def return_sub_num_only(sub_name):
    sub_nam = sub_name.split('/')[-1]
    sub_num = sub_nam.split('_')[2:5]
    join_str = '_'.join(e for e in sub_num)
    return join_str
def find_and_return_sub_path_in_next_step(sub_pth, m1, m2):
    sub_num = return_sub_num_only(sub_pth)
    x = sub_pth.replace(m1, m2)
    y = x.split('ADNI')[0]
    p = os.path.join(y, '*.*')
    for k in glob.glob(p):
        if str(sub_num) in k:
            y = k
            break
    return y
def get_images(dir_path, dir1, dir2, dir3, sav_vol_pth):
    dir_path = os.path.join(dir_path, '*.*')
    for indx, sub_pth in enumerate(glob.glob(dir_path)):
        bl_pths = save_images_return_paths_list(sub_pth, 110, dir1)
        path = find_and_return_sub_path_in_next_step(sub_pth, 'bl', 'm06')
        m06_pths = save_images_return_paths_list(path, 110, dir2)
        path = find_and_return_sub_path_in_next_step(sub_pth, 'bl', 'm12')
        m12_pths = save_images_return_paths_list(path, 110, dir3)
        all_imges_pth =  bl_pths + m06_pths + m12_pths
        volume = 0
        for indx, bl in enumerate(bl_pths):
            if indx == 0:
                volume = cv2.imread(bl, 0)
            else:
                im = cv2.imread(bl, 0)
                volume = np.dstack((volume, im))
        for m6 in m06_pths:
            img = cv2.imread(m6, 0)
            volume = np.dstack((volume, img))
        for m12 in m12_pths:
            img = cv2.imread(m12, 0)
            volume = np.dstack((volume, img))
        vol_save_pth = sav_vol_pth + '/' + sub_pth.split('/')[-1].replace('nii.gz', 'npy')
        np.save(vol_save_pth, volume)
        for z in all_imges_pth:
            os.remove(z)
        break

# Baseline directory path where volumes for CN or AD exists.
source_volumes= '/home/researchsrv1/Documents/nasir/Paper1/3DCNN/skull_striped_dataset/1.CN/bl'
# Distination directory where newley created volumes with selected slices are saved
resized_volume_save_path = '/home/researchsrv1/Documents/nasir/Paper1/3DCNN/skull_striped_dataset/1.CN/test'
# Directory path to save selected slices from the baseline volume
selected_slices_from_vol01 = '/home/researchsrv1/Documents/nasir/Paper1/3DCNN/skull_striped_dataset/1.CN/dr1'
# Directory path to save selected slices from the M06 volume
selected_slices_from_vol02 = '/home/researchsrv1/Documents/nasir/Paper1/3DCNN/skull_striped_dataset/1.CN/dr2'
# Directory path to save selected slices from the M12 volume
selected_slices_from_vol03 = '/home/researchsrv1/Documents/nasir/Paper1/3DCNN/skull_striped_dataset/1.CN/dr3'

get_images(source_volumes,
           selected_slices_from_vol01,
           selected_slices_from_vol02,
           selected_slices_from_vol03,
           resized_volume_save_path
           )
