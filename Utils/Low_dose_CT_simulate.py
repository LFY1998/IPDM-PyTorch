import functools
import os
from glob import glob
from multiprocessing import Pool
import torch
from Recon.FBP_kernel import FBP
from Recon.TASART2DNSL0 import recons_torch, proj_torch
import numpy as np
from tqdm import tqdm


# 低剂量CT模拟
def worker(patient_path, Dose):
    convertor = init_convertor(mode="ART")
    image_names = sorted(glob(patient_path + '/*'))
    proj_save_path_root = patient_path.replace("ND", "{}dose".format(Dose))
    img_save_path_root = patient_path.replace("ND", "{}dose".format(Dose)).replace("proj", "miu")
    os.makedirs(proj_save_path_root, exist_ok=True)
    os.makedirs(img_save_path_root, exist_ok=True)

    for proj_path in image_names:
        noise_img_save_path = img_save_path_root + "\\" + proj_path.split("\\")[-1]
        noise_proj_save_path = proj_save_path_root + "\\" + proj_path.split("\\")[-1]
        if os.path.exists(noise_img_save_path) and os.path.exists(noise_proj_save_path):
            continue
        # 读取二进制txt,转numpy，加噪
        try:
            clean_proj = np.load(proj_path)
            noise_proj = add_noise(clean_proj, Dose)
            noise_img = convertor[0](torch.from_numpy(noise_proj[None]).type(torch.float32))[0].cpu().numpy()
            np.save(noise_img_save_path, noise_img)
            np.save(noise_proj_save_path, noise_proj.astype(np.float32))
        except:
            print(proj_path)
            continue


def add_noise(data: np.array, factor=0.5):
    Ne = 5.8
    N0 = 1.4 * 1e5
    n = np.random.randn(data.shape[0], data.shape[1])
    noise_data = data + np.sqrt(
        (1 - factor) * np.exp(data) * (1 + ((1 + factor) * Ne * np.exp(data)) / (factor * N0)) / (factor * N0)) * n
    return noise_data


def ldct_simulate(data_dir, num_threads, dose):
    patient_names = sorted(glob(data_dir + '/*'))
    with Pool(processes=num_threads) as pool:
        for _ in tqdm(pool.imap(functools.partial(worker, dose=dose), patient_names),
                      total=len(patient_names)):
            pass


def init_convertor(mode, device="cuda:0"):
    sa = np.fromfile(r"Recon/Simens_alut.txt", "float32")
    st = np.fromfile(r"Recon/Simens_theta.txt", "float32")
    if mode == "FBP":
        recon = FBP(device=device).convert
    elif mode == "ART":
        recon = functools.partial(recons_torch, lut_area=sa, betas=st, nstart=10, ntv=0,
                                  sample_rate=1, permute=True)
    projector = functools.partial(proj_torch, lut_area=sa, betas=st)
    return recon, projector


if __name__ == '__main__':
    for dose in [0.25]:
        ldct_simulate('G:/ddpm_mayo/test/ND/proj_npz', 4, dose=dose)
