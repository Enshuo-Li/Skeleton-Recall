import os
os.environ['nnUNet_raw'] = r'/home/med_bci/lienshuo/nnUNet_raw/'
os.environ['nnUNet_preprocessed'] = r'/home/med_bci/lienshuo/nnUNet_preprocessed/'
os.environ['nnUNet_results'] = r'/home/med_bci/lienshuo/nnUNet_results/'
from glob import glob
from tqdm import tqdm
import shutil

import multiprocessing
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


#%%
def convert_ImageTBAD(imageTBAD_base_dir: str = r'/home/med_bci/lienshuo/data/ImageTBAD/',
                      nnunet_dataset_id: int = 1):
    task_name = "ImageTBAD"

    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    cases = subfiles(imageTBAD_base_dir, suffix='_image.nii.gz', join=False)
    for case in tqdm(cases):
        case_label = case.replace('_image.nii.gz', '_label.nii.gz')
        case_id = int(case.split('_')[0])

        shutil.copy(join(imageTBAD_base_dir, case), join(imagestr, f'ImageTBAD_{case_id:03d}_0000.nii.gz'))
        shutil.copy(join(imageTBAD_base_dir, case_label), join(labelstr, f'ImageTBAD_{case_id:03d}.nii.gz'))

    generate_dataset_json(out_base, {0: "CT"},  # CTA -> CT
                          labels={
                              "background": 0,
                              "TL": 1,
                              "FL": 2,
                              "FLT": 3,
                              "aorta": (1, 2, 3)
                          },
                          regions_class_order=(2, 3, 4, 1),
                          num_training_cases=len(cases), file_ending='.nii.gz',
                          dataset_name=task_name, reference='none',
                          overwrite_image_reader_writer='NibabelIOWithReorient',
                          description="ImageTBAD")


#%%
if __name__ == '__main__':
    convert_ImageTBAD()

# ImageTBAD_path = r'/home/med_bci/lienshuo/data/ImageTBAD/'
# nnunet_raw_path = r'/home/med_bci/lienshuo/nnUNet_raw/Dataset001_ImageTBAD/'
#
# image_path_list = glob(ImageTBAD_path + '*_image.nii.gz')
# label_path_list = glob(ImageTBAD_path + '*_label.nii.gz')
#
# for image_path in image_path_list:
#     shutil.copy(image_path, image_path.replace(ImageTBAD_path, nnunet_raw_path + 'imagesTr/'))
# for label_path in label_path_list:
#     shutil.copy(label_path, label_path.replace(ImageTBAD_path, nnunet_raw_path + 'labelsTr/'))
