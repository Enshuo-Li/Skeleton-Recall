import os
# 我已经把修改环境变量的内容添加到了.bashrc文件中
# os.environ['nnUNet_raw'] = r'/home/med_bci/lienshuo/nnUNet_raw/'
# os.environ['nnUNet_preprocessed'] = r'/home/med_bci/lienshuo/nnUNet_preprocessed/'
# os.environ['nnUNet_results'] = r'/home/med_bci/lienshuo/nnUNet_results/'
# 但是这种方式似乎只能影响到ssh中的环境变量，并不能影响到当前pycharm debug和console环境中的环境变量
# 因而在这个文件中我进行额外的手动添加
os.environ['nnUNet_raw'] = r'/home/public_workspace/workspace/ES_Li/nnUNet_raw/'
os.environ['nnUNet_preprocessed'] = r'/home/public_workspace/workspace/ES_Li/nnUNet_preprocessed/'
os.environ['nnUNet_results'] = r'/home/public_workspace/workspace/ES_Li/nnUNet_results/'
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
def convert_mha_to_nrrd(mha_file_path, nrrd_file_path):
    # 读取MHA文件
    image = sitk.ReadImage(mha_file_path, sitk.sitkFloat32)

    # 将读取的图像转换为NRRD格式并保存
    sitk.WriteImage(image, nrrd_file_path)

def convert_TopCoW(TopCoW_base_dir: str = r'/home/public_workspace/workspace/ES_Li/data/TopCoW2024_Data_Release/',
                   nnunet_dataset_id: int = 4,
                   task_name: str = "TopCoW"):

    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    ##########################下面代码需要修改###########################
    cases = subfiles(os.path.join(TopCoW_base_dir, 'imagesTr/'), prefix='topcow_ct_', suffix='.nii.gz', join=False)
    for case_ct in tqdm(cases):
        case_mr = case_ct.replace('topcow_ct_', 'topcow_mr_')
        case_ct_label = case_ct.replace('_0000.nii.gz', '.nii.gz')
        case_mr_label = case_mr.replace('_0000.nii.gz', '.nii.gz')
        case_id = int(case_ct.split('_')[2])

        shutil.copy2(join(TopCoW_base_dir, 'imagesTr', case_ct),
                     join(imagestr, f'{task_name}_{case_id:03d}_0000.nii.gz'))
        shutil.copy2(join(TopCoW_base_dir, 'cow_seg_labelsTr', case_ct_label),
                     join(labelstr, f'{task_name}_{case_id:03d}_0000.nii.gz'))
        shutil.copy2(join(TopCoW_base_dir, 'imagesTr', case_mr),
                     join(imagestr, f'{task_name}_{case_id:03d}_0001.nii.gz'))
        shutil.copy2(join(TopCoW_base_dir, 'cow_seg_labelsTr', case_mr_label),
                     join(labelstr, f'{task_name}_{case_id:03d}_0001.nii.gz'))


    generate_dataset_json(out_base, {0: "CT", 1: "MRI"},  # CTA -> CT
                          labels={
                              "background": 0,
                              "Zone_0": 1,
                              "Innominate": 2,
                              "Zone_1": 3,
                              "Left_Common_Carotid": 4,
                              "Zone_2": 5,
                              "Left_Subclavian_Artery": 6,
                              "Zone_3": 7,
                              "Zone_4": 8,
                              "Zone_5": 9,
                              "Zone_6": 10,
                              "Celiac_Artery": 11,
                              "Zone_7": 12,
                              "SMA": 13,
                              "Zone_8": 14,
                              "Right_Renal_Artery": 15,
                              "Left_Renal_Artery": 16,
                              "Zone_9": 17,
                              "Zone_10_R(Right_Common_Iliac_Artery)": 18,
                              "Zone_10_L(Left_Common_Iliac_Artery)": 19,
                              "Right_Internal_Iliac_Artery": 20,
                              "Left_Internal_Iliac_Artery": 21,
                              "Zone_11_R(Right_External_Iliac_Artery)": 22,
                              "Zone_11_L(Left_External_Iliac_Artery)": 23
                          },
                          # regions_class_order=(2, 3, 4, 1),
                          num_training_cases=len(cases), file_ending='.nii.gz',
                          dataset_name=task_name, reference='none',
                          # overwrite_image_reader_writer='NibabelIOWithReorient',
                          description="AortaSeg24")


#%%
if __name__ == '__main__':
    # convert_TopCoW(TopCoW_base_dir=r"/home/public_workspace/workspace/ES_Li/data/AortaSeg24/")
    convert_TopCoW(TopCoW_base_dir=r"E:\LES\data\TopCoW2024_Data_Release")

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
