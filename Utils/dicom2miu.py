import cv2
import numpy as np
import os
import functools
import PIL.Image as Image
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
import SimpleITK as sitk


# DICOM转μ
def worker(patient_path):
    image_names = sorted(glob(patient_path + '/*'))
    N = len(image_names)
    miu_water = 0.183
    txt_save_path = patient_path.replace("dcm", "miu_txt")
    if not os.path.exists(txt_save_path):
        os.makedirs(txt_save_path)
    npz_save_path = patient_path.replace("dcm", "miu_npz")
    if not os.path.exists(npz_save_path):
        os.makedirs(npz_save_path)
    # proj_save_path=patient_path.replace("dcm", "proj")

    for dcm in image_names:
        save_txt_path = txt_save_path + "\\" + dcm.split("\\")[-1] + ".txt"
        save_npz_path = npz_save_path + "\\" + dcm.split("\\")[-1] + ".npy"
        # 读取dcm，转成miu
        try:
            HU = sitk.GetArrayFromImage(sitk.ReadImage(dcm))[0] + 24
        except:
            print("读取失败:", dcm)
            continue
        #       转成二进制，存入对应的文件夹
        if HU.min() == HU.max() or HU.shape != (512, 512):
            continue
        else:
            miu = (miu_water + (HU * miu_water / 1e3)).astype(np.float32)
            miu.flatten('F').tofile(save_txt_path)
            np.save(save_npz_path, miu)


def create_miu_binary(data_dir, num_threads):
    patient_names = sorted(glob(data_dir + '/*'))
    with Pool(processes=num_threads) as pool:
        for _ in tqdm(pool.imap_unordered(functools.partial(worker), patient_names), total=len(patient_names)):
            pass


if __name__ == '__main__':
    create_miu_binary(r'F:\ddpm_mayo\test\test_dcm', 4)
