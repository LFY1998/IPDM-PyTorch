import glob
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor


def pixel2HU(img, window=None):
    if window is None:
        window = [-1024, 3072]
    return img * (window[1] - window[0]) + window[0]


def HU2miu(HU):
    miu_water = 0.183
    return miu_water + ((HU + 24) * miu_water / 1e3)


def miu2HU(miu):
    miu_water = 0.183
    return (miu - miu_water) * 1e3 / miu_water - 24


def miu2pixel(miu, HU_range=None):
    HU = miu2HU(miu)
    return HU2pixel(HU, HU_range)


def HU2pixel(HU, new_window=None):
    if new_window is None:
        new_window = [-1024, 3072]
    img = (HU - new_window[0]) / (new_window[1] - new_window[0])
    img[HU < new_window[0]] = 0
    img[HU > new_window[1]] = 1
    return img


def pixel2miu(pix):
    return HU2miu(pixel2HU(pix))


def reset_window_centre(img: np.ndarray, new_window=None, origin_window=None):
    if origin_window is None:
        origin_window = [-1024, 3072]
    if new_window is None:
        new_window = origin_window
    HU_ = (img * (origin_window[1] - origin_window[0]) + origin_window[0])
    img = (HU_ - new_window[0]) / (new_window[1] - new_window[0])
    img[HU_ < new_window[0]] = 0
    img[HU_ > new_window[1]] = 1
    return img


class Siemens_dataset_npz(Dataset):
    def __init__(self, ldproj_path=None, ldimg_path=None, fdproj_path=None, fdimg_path=None, proj_clip=False,
                 img_clip=True, data_type='siemens', patch=None, patch_per_image=None, assign=None):
        self.patch_per_image = patch_per_image
        self.patient_name = None
        self.slice_name = None
        self.data_type = data_type
        self.ldproj_path = ldproj_path
        self.ldimg_path = ldimg_path
        self.fdproj_path = fdproj_path
        self.fdimg_path = fdimg_path
        self.proj_clip = proj_clip
        self.img_clip = img_clip
        if fdimg_path is not None:
            self.fdimg_file_name = sorted(glob.glob(fdimg_path + "/*/*"))
            if assign is not None:
                self.fdimg_file_name = list(filter(lambda x: x.split('\\')[-2] in assign, self.fdimg_file_name))
            self.get_pname_sname(self.fdimg_file_name)
        if fdproj_path is not None:
            self.fdproj_file_name = sorted(glob.glob(fdproj_path + "/*/*"))
            if assign is not None:
                self.fdproj_file_name = list(filter(lambda x: x.split('\\')[-2] in assign, self.fdproj_file_name))
            self.get_pname_sname(self.fdproj_file_name)
        if ldimg_path is not None:
            self.ldimg_file_name = sorted(glob.glob(ldimg_path + "/*/*"))
            self.get_pname_sname(self.ldimg_file_name)
        if ldproj_path is not None:
            self.ldproj_file_name = sorted(glob.glob(ldproj_path + "/*/*"))
            self.get_pname_sname(self.ldproj_file_name)

        self.patch = patch

    def __getitem__(self, idx):
        return_data = [None, None, None, None]
        if self.ldimg_path is not None:
            ldimg = self.get_data(self.ldimg_file_name[idx])
            if self.patch is not None:
                return_data[0] = self.get_patch(ToTensor()(ldimg))
            else:
                return_data[0] = ToTensor()(ldimg)
        if self.fdproj_path is not None:
            fdproj = self.get_data(self.fdproj_file_name[idx])
            if self.proj_clip:
                fdproj = fdproj / 10
            if self.patch is not None:
                return_data[1] = self.get_patch(ToTensor()(fdproj))
            else:
                return_data[1] = ToTensor()(fdproj)
        if self.fdimg_path is not None:
            fdimg = self.get_data(self.fdimg_file_name[idx])
            if self.patch is not None:
                return_data[2] = self.get_patch(ToTensor()(fdimg))
            else:
                return_data[2] = ToTensor()(fdimg)
        if self.ldproj_path is not None:
            ldproj = self.get_data(self.ldproj_file_name[idx])
            if self.proj_clip:
                ldproj = ldproj / 10
            if self.patch is not None:
                return_data[3] = self.get_patch(ToTensor()(ldproj))
            else:
                return_data[3] = ToTensor()(ldproj)
        return return_data

    def get_pname_sname(self, file_path):
        if self.patient_name is None:
            if self.data_type == "siemens":
                self.patient_name = list(map(lambda x: x.split("\\")[-2], file_path))
                self.slice_name = list(map(lambda x: x.split("\\")[-1].split(".")[0], file_path))
            elif self.data_type == "mayo":
                self.patient_name = list(map(lambda x: x.split("\\")[-2], file_path))
                self.slice_name = list(map(lambda x: x.split("\\")[-1].split(".")[-4], file_path))

    @staticmethod
    def get_data(file_path):
        if file_path.split(".")[-1] == "npz":
            img = np.load(file_path)["arr_0"]
        else:
            img = np.load(file_path)
        return img

    def __len__(self):
        if self.fdimg_path is not None:
            return len(self.fdimg_file_name)
        if self.fdproj_path is not None:
            return len(self.fdproj_file_name)
        if self.ldimg_path is not None:
            return len(self.ldimg_file_name)
        if self.ldproj_path is not None:
            return len(self.ldproj_file_name)

    def get_data_from_name(self, patient_name, slice_name):
        return_data = [None, None, None, None]
        if self.ldimg_path is not None:
            ldimg_path = list(filter(lambda x: patient_name in x and slice_name in x, self.ldimg_file_name))[0]
            ldimg = self.get_data(ldimg_path)
            return_data[0] = ToTensor()(ldimg)
        if self.fdproj_path is not None:
            fdproj_path = list(filter(lambda x: patient_name in x and slice_name in x, self.fdproj_file_name))[0]
            fdproj = self.get_data(fdproj_path)
            if self.proj_clip:
                fdproj = fdproj / 10
            return_data[1] = ToTensor()(fdproj)
        if self.fdimg_path is not None:
            fdimg_path = list(filter(lambda x: patient_name in x and slice_name in x, self.fdimg_file_name))[0]
            fdimg = self.get_data(fdimg_path)
            return_data[2] = ToTensor()(fdimg)
        if self.ldproj_path is not None:
            ldproj_path = list(filter(lambda x: patient_name in x and slice_name in x, self.ldproj_file_name))[0]
            ldproj = self.get_data(ldproj_path)
            if self.proj_clip:
                ldproj = ldproj / 10
            return_data[3] = ToTensor()(ldproj)
        return return_data

    def get_patch(self, data):
        if self.patch is not None:
            patch = torch.zeros((self.patch_per_image, self.patch[0], self.patch[1]))
            for i in range(self.patch_per_image):
                seed = torch.random.seed()
                torch.random.manual_seed(seed)
                patch[i, :, :] = transforms.RandomCrop(self.patch)(data)
            return patch

    @staticmethod
    def collate(batch_data):
        ld_img = [item[0] for item in batch_data]
        if ld_img[0] is not None:
            ld_img = torch.stack(ld_img, dim=0)
        else:
            ld_img = None
        fd_proj = [item[1] for item in batch_data]
        if fd_proj[0] is not None:
            fd_proj = torch.stack(fd_proj, dim=0)
        else:
            fd_proj = None
        fd_img = [item[2] for item in batch_data]
        if fd_img[0] is not None:
            fd_img = torch.stack(fd_img, dim=0)
        else:
            fd_img = None
        ld_proj = [item[3] for item in batch_data]
        if ld_proj[0] is not None:
            ld_proj = torch.stack(ld_proj, dim=0)
        else:
            ld_proj = None
        return ld_img, fd_proj, fd_img, ld_proj
