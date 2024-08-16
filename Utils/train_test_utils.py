import copy
import functools
import json
import os
import os.path as osp
from typing import Optional

import torch
from torch import Tensor
import torch.distributed as dist
import tqdm
from datetime import datetime

from Utils.NQM import NQM
from Model.model import UNetModel, GaussianDiffusion, yeo_johnson_transform
import numpy as np
from matplotlib import pyplot as plt
from Recon.FBP_kernel import FBP
from Recon.TASART2DNSL0 import recons_torch, proj_torch
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.utils.tensorboard import SummaryWriter

from Config.default_config import cfg_load
from Utils.loggerx import LoggerX
from Dataset.npz_data_loader import miu2pixel, Siemens_dataset_npz
from Utils.sampler import RandomSampler
from piq import vif_p, fsim


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value


class ResultTempDict(DotDict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getitem__(self, item):
        if isinstance(item, str):
            return super().__getitem__(item)
        elif isinstance(item, int):
            if item > 0:
                return self[f"iter_{item}"]
            elif item == -1:
                return self[f"iter_{len(self)}"]


def dict_add(d1, d2, d):
    """
    :param d1: total dict
    :param d2: instance dict
    :param d: 记录每条键值对加法的次数
    :return:
    """
    for key in d2.keys():
        if isinstance(d2[key], dict):
            if key not in d1.keys():
                d1[key] = dict()
                d[key] = dict()
            dict_add(d1[key], d2[key], d[key])
        else:
            if key not in d1.keys():
                d1[key] = 0
                d[key] = 0
            d1[key] += d2[key]
            d[key] += 1


def dict_mean(d1, d):
    """
    :param d1: total dict
    :param d: 记录每条值需要除的次数的dict
    :return:
    """
    for key in d1.keys():
        if isinstance(d1[key], dict):
            dict_mean(d1[key], d[key])
        else:
            d1[key] /= d[key]


def dict_value_minus_mean_square(d1, d_mean, d):
    for key in d1.keys():
        if isinstance(d1[key], dict):
            if key not in d.keys():
                d[key] = dict()
            dict_value_minus_mean_square(d1[key], d_mean[key], d[key])
        else:
            if key + "_std" not in d_mean.keys():
                d_mean[key + "_std"] = 0
                # 样本标准差除以N-1，初始值设为-1,除以N，则初始值设为0
                d[key + "_std"] = 0
            d_mean[key + "_std"] += (d1[key] - d_mean[key]) ** 2
            d[key + "_std"] += 1


def dict_std(d1, d):
    for key in d1.keys():
        if isinstance(d1[key], dict):
            dict_std(d1[key], d[key])
        else:
            if "std" in key:
                if d[key] >= 1:
                    d1[key] = (d1[key] / d[key]) ** 0.5
                else:
                    d1[key] = 0
    return d1


class progressive_domain_denoiser:
    def __init__(self, opt, result_save_path=None):
        # Section: load option
        self.trans_ldproj = None
        self.trans_ldimg = None
        self.opt = opt
        self.opt_temp = copy.deepcopy(opt)
        timestamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        if result_save_path is None:
            save_root = osp.join(osp.dirname(osp.abspath(__file__)), 'ModelTrainLog/',
                                 '{}_{}/{}'.format(opt.model_name, opt.run_name, timestamp))
        else:
            save_root = osp.join(result_save_path, '{}_{}'.format(opt.model_name, opt.run_name))
        self.logger = LoggerX(save_root, opt)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.logger.save_option(self.opt)
        if "train" in self.opt.mode:
            self.summer = SummaryWriter(log_dir=save_root + '/trainSummary')
        for k in opt.__dict__:
            print(k + ": " + str(opt.__dict__[k]))

        self.optimizer = None
        self.proj_model = None
        self.img_model = None
        # Section: Projection Domain Denoising Model
        if self.opt.mode in ["train_proj", "test_proj", "test_prog"]:
            self.init_proj_model()
            if self.opt.mode == "train_proj":
                self.train_model = self.proj_model
                self.optimizer = torch.optim.Adam(self.train_model.parameters(), opt.init_lr, weight_decay=1e-5,
                                                  betas=(0.9, 0.999))
                self.partial_timesteps = self.opt.partial_timesteps_proj
                self.gaussian_diffusion_train = self.proj_gaussian_diffusion
                self.train_resume_epochs = self.opt.resume_epochs_proj

        # Section:Domain convertor
        self.init_convertor(opt.convertor)

        # Section: Image Domain Denoising Model
        if self.opt.mode in ["train_img", "test_img", "test_prog"]:
            self.init_img_model()
            if self.opt.mode == "train_img":
                self.train_model = self.img_model
                self.optimizer = torch.optim.Adam(self.train_model.parameters(), opt.init_lr, weight_decay=1e-5,
                                                  betas=(0.9, 0.999))
                self.partial_timesteps = self.opt.partial_timesteps_img
                self.gaussian_diffusion_train = self.img_gaussian_diffusion
                self.train_resume_epochs = self.opt.resume_epochs_img
        self.logger.modules = [self.proj_model, self.img_model, self.optimizer]
        self.logger.module_names = ["proj_model", "img_model", "optimizer"]
        self.load_model()

        # Section:data temp
        self.init_data_loader()
        self.fdct = None
        self.fdproj = None
        self.ldct = None
        self.ldct_np = None
        self.ldproj = None
        self.ldproj_np = None

        # Section:result temp
        self.proj_denoise_result = ResultTempDict()
        self.proj_denoise_convert2img_result = ResultTempDict()
        self.img_denoise_result = ResultTempDict()
        self.progressive_denoise_result = ResultTempDict()
        self.noise_strength = None
        # Section:condition curve initialize
        self.img_lambda_curve = curve_init()
        self.proj_lambda_curve = proj_curv_init()
        # Section: metric initialize
        self.metric_instance = DotDict(
            LDCT=DotDict(), deProj=DotDict(), deImg=DotDict(), deProg=DotDict(), deProj2img=DotDict())
        self.metric_total = DotDict()
        self.metric_each_sample = []

        # Section:test result and metric save path
        self.save_root_path = osp.join(save_root, 'save_test_results')
        if not os.path.exists(self.save_root_path):
            os.makedirs(self.save_root_path)

    def update_opt(self, ultra_cfg=None):
        # 合并cfg
        if ultra_cfg is not None:
            cfg_load(ultra_cfg, self.opt.__dict__)
            self.logger.save_option(self.opt)
        if "convertor" in ultra_cfg.keys():
            self.init_convertor(ultra_cfg["convertor"])

    def reset_opt(self):
        self.opt = copy.deepcopy(self.opt_temp)

    def init_img_model(self):
        self.img_model = UNetModel(in_channels=self.opt.in_channels_img,
                                   model_channels=self.opt.model_channels_img,
                                   out_channels=self.opt.out_channels_img,
                                   attention_resolutions=self.opt.attention_resolutions_img,
                                   channel_mult=self.opt.channel_mult_img).to(self.opt.device)
        self.img_device = next(self.img_model.parameters()).device
        self.img_dtype = next(self.img_model.parameters()).dtype
        self.img_gaussian_diffusion = GaussianDiffusion(timesteps=self.opt.timesteps_img,
                                                        beta_schedule='cosine',
                                                        schedule_power=self.opt.schedule_power_img)

    def init_convertor(self, convertor):
        sa = np.fromfile(r"Recon/Simens_alut.txt", "float32")
        st = np.fromfile(r"Recon/Simens_theta.txt", "float32")
        if convertor == "FBP":
            self.convertor = FBP(device=self.opt.device).convert
        elif convertor == "ART":
            self.convertor = functools.partial(recons_torch, lut_area=sa, betas=st, nstart=10, ntv=self.opt.ntv,
                                               sample_rate=1, permute=True)
        self.projection = functools.partial(proj_torch, lut_area=sa, betas=st)

    def init_proj_model(self):
        self.proj_model = UNetModel(in_channels=self.opt.in_channels_proj,
                                    model_channels=self.opt.model_channels_proj,
                                    out_channels=self.opt.out_channels_proj,
                                    attention_resolutions=self.opt.attention_resolutions_proj,
                                    channel_mult=self.opt.channel_mult_proj).to(self.opt.device)
        self.proj_device = self.opt.device
        self.proj_dtype = next(self.proj_model.parameters()).dtype
        self.proj_gaussian_diffusion = GaussianDiffusion(timesteps=self.opt.timesteps_proj,
                                                         beta_schedule='cosine',
                                                         schedule_power=self.opt.schedule_power_proj)

    def load_model(self):
        if self.opt.resume_epochs_img > 0 and self.opt.load_img_model_path is not None and self.img_model is not None:
            self.logger.load_checkpoints(self.opt.resume_epochs_img, self.opt.load_img_model_path)
        if self.opt.resume_epochs_proj > 0 and self.opt.load_proj_model_path is not None and self.proj_model is not None:
            self.logger.load_checkpoints(self.opt.resume_epochs_proj, self.opt.load_proj_model_path)

    def train(self, images, n_iter, loss_temp):
        self.train_model.train()
        self.optimizer.zero_grad()
        if self.opt.mode == 'train_proj':
            images = images[1]
        elif self.opt.mode == 'train_img':
            images = images[2]
        images = images.view(images.shape[0] * images.shape[1], 1, images.shape[2], -1)
        bs = images.shape[0]
        images = images.float().to(self.opt.device).clamp(min=0)
        if self.opt.normal:
            images, _ = yeo_johnson_transform(images)
        t = torch.randint(0, self.partial_timesteps, (bs,), device=self.opt.device).long()
        loss = self.gaussian_diffusion_train.train_losses(self.train_model, images, t)
        loss.backward()
        self.optimizer.step()
        lr = self.optimizer.param_groups[0]['lr']
        loss = loss.item()
        loss_temp[0] += loss
        self.logger.msg([loss, lr], n_iter)

    @torch.no_grad()
    def test(self, epoch):
        if self.opt.mode in ["train_proj", "test_proj"]:
            self.proj_model.eval()
        if self.opt.mode in ["train_img", "test_img"]:
            self.img_model.eval()
        if self.opt.mode == "test_prog":
            self.proj_model.eval()
            self.img_model.eval()
        if self.opt.test_numbers <= 0:
            self.opt.test_numbers = len(self.test_dataset)
        np.random.seed(9527)
        random_test_id = np.sort(np.random.choice(len(self.test_dataset), self.opt.test_numbers, replace=False))
        with tqdm.tqdm(initial=0, total=self.opt.test_numbers + 1, desc='test_process.....') as T2:
            # for idx, image in enumerate(self.test_loader):
            for idx in range(0, self.opt.test_numbers):
                ld_img, fd_proj, fd_img, ld_proj = self.test_dataset[random_test_id[idx]]
                try:
                    ld_img, fd_img, ld_proj = ld_img[None], fd_img[None], ld_proj[None]
                except:
                    ld_img, fd_img = ld_img[None], fd_img[None]
                self.temp_clear()
                self.metric_clear()
                self.save_path_load(epoch, self.test_dataset.patient_name[random_test_id[idx]],
                                    self.test_dataset.slice_name[random_test_id[idx]])

                self.data_sample_load(ldct=ld_img, ldproj=ld_proj, fdproj=fd_proj, fdct=fd_img)
                if self.opt.mode in ["train_proj", "test_proj"]:
                    _ = self.proj_denoiser(self.ldproj)
                    self.result_figure_save(mode="dproj2img", display=False, only_metric=not self.opt.display_result)
                if self.opt.mode in ["train_img", "test_img"]:
                    _ = self.img_denoiser(self.ldct, mode="img_only")
                    self.result_figure_save(mode="dimg", display=False, only_metric=not self.opt.display_result)
                if self.opt.mode == "test_prog":
                    _ = self.progressive_denoiser()
                    self.result_figure_save(mode="progressive", display=False, only_metric=not self.opt.display_result)
                self.result_data_save(data_save=self.opt.test_result_data_save)
                self.metric_update()
                T2.update(1)
                if idx == self.opt.test_numbers - 1:
                    break
        self.metric_total_save(epoch)
        if 'train' in self.opt.mode:
            # 如果在训练阶段，绘制指标变化图，首先检索不为空的指标
            for key in self.metric_total.keys():
                if self.metric_total[key]:
                    # 将字典中不同指标分开，存入summer
                    psnr_dict = {k: v for k, v in self.metric_total[key].items() if "psnr" in k}
                    self.summer.add_scalars(key + "/psnr", psnr_dict, global_step=epoch)
                    ssim_dict = {k: v for k, v in self.metric_total[key].items() if "ssim" in k}
                    self.summer.add_scalars(key + "/ssim", ssim_dict, global_step=epoch)

    def fit(self):
        opt = self.opt
        if 'train' in opt.mode:
            # training routine
            loader = iter(self.train_loader)
            loss_temp = [0]
            for n_iter in tqdm.trange(opt.resume_iter + 1, opt.max_iter + 1, disable=(self.rank != 0)):
                inputs = next(loader)
                # 每 10 步 统计一次loss均值
                self.train(inputs, n_iter, loss_temp)
                if n_iter % 10 == 0:
                    loss_mean = loss_temp[0] / 10
                    #     更新绘制损失图
                    self.summer.add_scalar("train/loss", loss_mean, global_step=n_iter // 10)
                    loss_temp = [0]
                if n_iter % self.opt.save_freq == 0:
                    it = n_iter // self.opt.save_freq
                    self.logger.checkpoints(it)
                    if self.opt.test_numbers > 0:
                        self.test(it)

        elif 'test' in opt.mode:
            self.test(0)

    def init_data_loader(self):
        if 'train' in self.opt.mode:
            train_dataset = Siemens_dataset_npz(ldimg_path=self.opt.train_dataset_path_LD_img,
                                                fdimg_path=self.opt.train_dataset_path_FD_img,
                                                ldproj_path=self.opt.train_dataset_path_LD_proj,
                                                fdproj_path=self.opt.train_dataset_path_FD_proj,
                                                proj_clip=self.opt.clip_proj,
                                                img_clip=self.opt.clip_img,
                                                data_type=self.opt.data_type,
                                                patch=self.opt.patch,
                                                patch_per_image=self.opt.patch_per_image)
            self.opt.max_iter = len(train_dataset) * self.opt.max_epochs // self.opt.batch_size
            self.opt.resume_iter = self.train_resume_epochs * self.opt.save_freq // self.opt.batch_size
            train_sampler = RandomSampler(dataset=train_dataset, batch_size=self.opt.batch_size,
                                          num_iter=self.opt.max_iter, restore_iter=self.opt.resume_iter)
            self.train_len = len(train_dataset)
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=self.opt.batch_size,
                sampler=train_sampler,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                collate_fn=train_dataset.collate,
                # num_workers=self.opt.num_workers
            )
            self.train_loader = train_loader
        test_dataset = Siemens_dataset_npz(ldimg_path=self.opt.test_dataset_path_LD_img,
                                           fdimg_path=self.opt.test_dataset_path_FD_img,
                                           ldproj_path=self.opt.test_dataset_path_LD_proj,
                                           fdproj_path=self.opt.test_dataset_path_FD_proj,
                                           proj_clip=self.opt.clip_proj,
                                           img_clip=self.opt.clip_img,
                                           data_type=self.opt.data_type,
                                           patch=None,
                                           patch_per_image=None)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=self.opt.test_batch_size,
            shuffle=False,
            collate_fn=test_dataset.collate,
            # num_workers=self.opt.num_workers
        )
        self.test_loader = test_loader

        self.test_dataset = test_dataset

    def temp_clear(self):
        self.proj_temp_clear()
        self.img_temp_clear()
        self.metric_clear()
        # self.fdct = None
        # self.fdproj = None
        # self.ldct = None
        # self.ldct_np = None
        # self.ldproj = None
        # self.ldproj_np = None
        self.noise_strength = None

    def metric_clear(self):
        self.metric_instance = DotDict(LDCT=DotDict(), deProj=DotDict(), deImg=DotDict(), deProg=DotDict(),
                                       deProj2img=DotDict())

    def proj_temp_clear(self):
        self.proj_denoise_convert2img_result = ResultTempDict()
        self.proj_denoise_result = ResultTempDict()

    def img_temp_clear(self):
        self.img_denoise_result = ResultTempDict()
        self.progressive_denoise_result = ResultTempDict()

    def proj_denoiser(self, x: Tensor, convert=True, save_state=True, save_proj_state=False, return_idx=-1):
        # save_state:是否保留中间状态，如果convert为TRue,则保留的状态为重建图，否则为投影图，如果不保留，则只返回最后的去噪结果
        # save_proj_state：可以单独设置投影图去噪结果的保留与否
        # print("pd start...")

        with torch.no_grad():
            if self.opt.sample_method_proj == "dense":
                result, _, noise_strength = self.proj_gaussian_diffusion.guided_reverse_process(
                    model=self.proj_model,
                    img=x.type(self.proj_dtype),
                    t_start=self.opt.t_start_proj,
                    clip=self.opt.clip_proj,
                    lambda_ratio=self.opt.lambda_ratio_proj,
                    eta=self.opt.eta_proj,
                    lambda_curve=self.proj_lambda_curve,
                    mode="proj",
                    constant_guidance=self.opt.constant_guidance_proj,
                    kernel_size_proj=self.opt.kernel_size_proj,
                    amplitude_proj=self.opt.amplitude_proj,
                    only_convertor=self.opt.benchmark_test,
                    normal=self.opt.normal,
                    transformer=self.trans_ldproj
                )
                self.noise_strength = noise_strength
            elif self.opt.sample_method_proj == "sparse":
                result = self.proj_gaussian_diffusion.sparse_guided_reverse_process(model=self.proj_model,
                                                                                    condition=x.type(self.proj_dtype),
                                                                                    t_start=self.opt.t_start_proj,
                                                                                    condition_lambda_max=0.49,
                                                                                    condition_lambda_min=0.35,
                                                                                    clip_denoised=self.opt.clip_proj,
                                                                                    ddim_timesteps=self.opt.ddim_timesteps_proj,
                                                                                    eta=self.opt.eta_proj)
        self.proj_temp_clear()
        if self.opt.clip_proj:
            G = 10
        else:
            G = 1
        if save_proj_state:
            for iters in range(len(result)):
                self.proj_denoise_result[f"iter_{iters + 1}"] = result[iters].cpu().numpy()
        if save_state:
            if convert:
                for iters in range(len(result)):
                    self.proj_denoise_convert2img_result[f"iter_{iters + 1}"] = self.convertor(
                        G * result[iters][:, 0, :, :]).unsqueeze(1).cpu().numpy()
                return torch.from_numpy(
                    self.proj_denoise_convert2img_result[f"iter_{len(result)}"]), self.noise_strength
            else:
                for iters in range(len(result)):
                    self.proj_denoise_result[f"iter_{iters + 1}"] = result[iters].cpu().numpy()
                return result[return_idx], self.noise_strength
        else:
            if convert:
                self.proj_denoise_convert2img_result["iter_1"] = self.convertor(
                    G * result[return_idx][:, 0, :, :].float()).unsqueeze(1).cpu().numpy()
                return torch.from_numpy(self.proj_denoise_convert2img_result["iter_1"]), self.noise_strength
            else:
                self.proj_denoise_result["iter_1"] = result[return_idx].cpu().numpy()
                return result[return_idx], self.noise_strength

    def img_denoiser(self, x, return_idx=-1, noise_strength=None, mode="progressive", sharpen_num=45, save_state=True):
        with torch.no_grad():
            if self.opt.sample_method_img == "dense":
                result, _, _ = self.img_gaussian_diffusion.guided_reverse_process(
                    model=self.img_model,
                    img=x.type(self.img_dtype).to(self.img_device),
                    t_start=self.opt.t_start_img,
                    clip=self.opt.clip_img,
                    # lambda_ratio:用于预去噪的条件比例，越大则去噪越少保守
                    lambda_ratio=self.opt.lambda_ratio_img,
                    eta=self.opt.eta_img,
                    # save_states:保留反向过程的每一步
                    save_states=self.opt.save_states_img,
                    lambda_curve=self.img_lambda_curve,
                    noise_strength=noise_strength,
                    ldct=x.type(self.img_dtype).to(self.img_device),
                    constant_guidance=self.opt.constant_guidance_img,
                    kernel_size_img=self.opt.kernel_size_img,
                    amplitude_img=self.opt.amplitude_img,
                    only_convertor=self.opt.benchmark_test,
                    normal=self.opt.normal,
                    transformer=self.trans_ldimg
                )
            elif self.opt.sample_method_img == "sparse":
                result = self.img_gaussian_diffusion.sparse_guided_reverse_process(model=self.img_model,
                                                                                   condition=x.type(self.img_dtype).to(
                                                                                       self.img_device),
                                                                                   t_start=self.opt.t_start_img,
                                                                                   condition_lambda_max=0.5,
                                                                                   condition_lambda_min=0.3,
                                                                                   clip_denoised=True,
                                                                                   ddim_timesteps=self.opt.ddim_timesteps_img,
                                                                                   eta=self.opt.eta_img)
            if self.opt.ultra_img_denoise:
                result_, _, _ = self.img_gaussian_diffusion.guided_reverse_process(
                    model=self.img_model,
                    img=result[-1],
                    t_start=[5, 5, 5],
                    clip=self.opt.clip_img,
                    # lambda_ratio:用于预去噪的条件比例，越大则去噪越少保守
                    lambda_ratio=self.opt.lambda_ratio_img,
                    eta=0.6,
                    # save_states:保留反向过程的每一步
                    save_states=self.opt.save_states_img,
                    lambda_curve=self.img_lambda_curve,
                    noise_strength=noise_strength,
                    ldct=x.type(self.img_dtype).to(self.img_device),
                    constant_guidance=0.6,
                    kernel_size_img=self.opt.kernel_size_img,
                    amplitude_img=self.opt.amplitude_img,
                    only_convertor=self.opt.benchmark_test,
                    normal=self.opt.normal,
                    transformer=self.trans_ldimg
                )
                result += result_
        self.img_temp_clear()
        if save_state:
            for iters in range(len(result)):
                if mode == "progressive":
                    self.progressive_denoise_result[f"iter_{iters + 1}"] = result[iters].cpu().numpy()
                elif mode == "img_only":
                    self.img_denoise_result[f"iter_{iters + 1}"] = result[iters].cpu().numpy()
        else:
            if mode == "progressive":
                self.progressive_denoise_result[f"iter_1"] = result[return_idx].cpu().numpy()
            elif mode == "img_only":
                self.img_denoise_result[f"iter_1"] = result[return_idx].cpu().numpy()

        return result[return_idx]

    def progressive_denoiser(self, save_proj_state=False, convert=True, sharpen_num=42):
        result, n_s = self.proj_denoiser(self.ldproj, save_state=self.opt.save_it_state_proj,
                                         save_proj_state=save_proj_state,
                                         convert=convert)
        if self.opt.convertor == "FBP" and self.opt.fbp_sharpen:
            sharpen_num = sharpen_num
        else:
            sharpen_num = -1
        if self.opt.normal:
            x, trans = yeo_johnson_transform(tensor_sharpen(result, sharpen_num))
            self.trans_ldimg = trans
        else:
            x = tensor_sharpen(result, sharpen_num)
        result = self.img_denoiser(x, noise_strength=n_s,
                                   save_state=self.opt.save_it_state_img)
        return result

    def data_sample_load(self, ldct=Optional[Tensor], ldproj=Optional[Tensor], fdproj=Optional[np.ndarray],
                         fdct=Optional[np.ndarray]):
        """
        @param ldct: miu tensor [1,1,512,512]
        @param ldproj: proj tensor [1,1,2000,912]
        @param fdproj: proj tensor [1,1,2000,912]
        @param fdct: miu tensor [1,1,512,512]
        """
        if ldct is not None:
            if self.opt.normal:
                ldct_norm, self.trans_ldimg = yeo_johnson_transform(ldct)
                self.ldct = ldct_norm.to(self.opt.device)
            else:
                self.ldct = ldct.to(self.opt.device)
            self.ldct_np = miu2pixel(ldct.squeeze().cpu().numpy())
        if ldproj is not None:
            if self.opt.normal:
                ldproj_norm, self.trans_ldproj = yeo_johnson_transform(ldproj)
                self.ldproj = ldproj_norm.to(self.opt.device)
            else:
                self.ldproj = ldproj.to(self.opt.device)
            self.ldproj_np = ldproj.squeeze().cpu().numpy()
        if fdct is not None:
            self.fdct = miu2pixel(fdct).squeeze().numpy()
        if fdproj is not None:
            self.fdproj = fdproj.squeeze().numpy()

    def result_figure_save(self, mode="progressive", display=True, only_metric=False):
        """
        @param display:
        @param mode: "progressive":dproj->dimg,dual domain denoised
                     "dproj":only proj domain denoised
                     "dimg":only img domain denoised
                     "dproj2img":show dproj convert to img
        """
        try:
            if mode not in ["progressive", "dimg", "dproj", "dproj2img"]:
                raise Exception("ValueError:mode should be one of: \"progressive\",\"dimg\",\"dproj\",\"dproj2img\"")
        except Exception as e:
            print(e)
            return -1
        fig = None
        save_path = self.save_path
        # Note: mode="proj", show "the difference between fdct and ldct" and "the difference between fdct and deproj"
        if mode == "dproj":
            delta_target = np.abs(self.fdproj - self.ldproj_np)
            delta_proj = []
            for i in range(1, len(self.proj_denoise_result) + 1):
                delta_proj.append(np.abs(self.proj_denoise_result[f"iter_{i}"][0, 0] - self.fdproj))
            fig, ax = plt.subplots(1, 1 + len(self.proj_denoise_result), figsize=(30, 30))

            max_ = delta_target.max()
            min_ = delta_target.min()
            ax[0].set_title("res target", fontsize=35, y=1.02)
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[0].imshow(delta_target, "inferno", vmin=min_, vmax=max_)

            for i, dp in enumerate(delta_proj):
                ax[i + 1].set_title(f"deProj iter{i + 1}", fontsize=35, y=1.02)
                ax[i + 1].set_xticks([])
                ax[i + 1].set_yticks([])
                ax[i + 1].imshow(dp, "inferno", vmin=min_, vmax=max_)
            plt.savefig(save_path + "/dProj.png", dpi=100)

        if mode == "dproj2img":
            self.metric_calculate(mode="LDCT", it=0, denoise_result=self.ldct_np)
            if not only_metric:
                fig, ax = plt.subplots(1, 2 + len(self.proj_denoise_convert2img_result),
                                       figsize=(7 * (2 + len(self.proj_denoise_convert2img_result)), 7))
                # LDCT
                ax[0].set_title("LDCT", fontsize=35, y=1.02)
                s = 'PSNR={:.2f}'.format(self.metric_instance.LDCT.psnr_iter_0) + ' , ' + 'SSIM={:.2f}'.format(
                    self.metric_instance.LDCT.ssim_iter_0)
                ax[0].text(x=0.5, y=-0.15, s=s, fontsize=25, horizontalalignment='center', transform=ax[0].transAxes)
                ax[0].set_xticks([])
                ax[0].set_yticks([])
                ax[0].imshow(self.ldct_np, "gray", vmin=(-160 + 1024) / 4096, vmax=(240 + 1024) / 4096)

                ax[1].set_title("FDCT", fontsize=35, y=1.02)
                ax[1].set_xticks([])
                ax[1].set_yticks([])
                ax[1].imshow(self.fdct, "gray", vmin=(-160 + 1024) / 4096, vmax=(240 + 1024) / 4096)

            img_its = len(self.proj_denoise_convert2img_result)
            for i in range(1, img_its + 1):
                r_it = img_its + 1 - i
                denoise_result = miu2pixel(self.proj_denoise_convert2img_result[f"iter_{r_it}"][0, 0])
                self.metric_calculate(mode="deProj2img", it=r_it, denoise_result=denoise_result)
                if not only_metric:
                    ax[i + 1].set_title("Proj iter{}".format(r_it), fontsize=35, y=1.02)
                    s = 'PSNR={:.2f}'.format(
                        self.metric_instance.deProj2img[f"psnr_iter_{r_it}"]) + ' , ' + 'SSIM={:.2f}'.format(
                        self.metric_instance.deProj2img[f"ssim_iter_{r_it}"])
                    ax[i + 1].text(s=s, fontsize=25, x=0.5, y=-0.15, horizontalalignment='center',
                                   transform=ax[i + 1].transAxes)
                    ax[i + 1].set_xticks([])
                    ax[i + 1].set_yticks([])
                    ax[i + 1].imshow(denoise_result, "gray", vmin=(-160 + 1024) / 4096, vmax=(240 + 1024) / 4096)
            if not only_metric:
                plt.savefig(save_path + "/deProj2img.png", dpi=200)

        # Note: mode="dimg",show "dimg" and "fdct" and "ldct"
        if mode == "dimg":
            self.metric_calculate(mode="LDCT", it=0, denoise_result=self.ldct_np)
            if not only_metric:
                fig, ax = plt.subplots(1, 2 + len(self.img_denoise_result),
                                       figsize=(7 * (2 + len(self.img_denoise_result)), 7))
                # LDCT
                ax[0].set_title("LDCT", fontsize=35, y=1.02)
                s = 'PSNR={:.2f}'.format(self.metric_instance.LDCT.psnr_iter_0) + ' , ' + 'SSIM={:.2f}'.format(
                    self.metric_instance.LDCT.ssim_iter_0)
                ax[0].text(x=0.5, y=-0.15, s=s, fontsize=25, horizontalalignment='center', transform=ax[0].transAxes)
                ax[0].set_xticks([])
                ax[0].set_yticks([])
                ax[0].imshow(self.ldct_np, "gray", vmin=(-160 + 1024) / 4096, vmax=(240 + 1024) / 4096)

                ax[1].set_title("FDCT", fontsize=35, y=1.02)
                ax[1].set_xticks([])
                ax[1].set_yticks([])
                ax[1].imshow(self.fdct, "gray", vmin=(-160 + 1024) / 4096, vmax=(240 + 1024) / 4096)

            img_its = len(self.img_denoise_result)
            for i in range(1, img_its + 1):
                r_it = img_its + 1 - i
                denoise_result = miu2pixel(self.img_denoise_result[f"iter_{r_it}"][0, 0])
                self.metric_calculate(mode="deImg", it=r_it, denoise_result=denoise_result)
                if not only_metric:
                    ax[i + 1].set_title("Img iter{}".format(r_it), fontsize=35, y=1.02)
                    s = 'PSNR={:.2f}'.format(
                        self.metric_instance.deImg[f"psnr_iter_{r_it}"]) + ' , ' + 'SSIM={:.2f}'.format(
                        self.metric_instance.deImg[f"ssim_iter_{r_it}"])
                    ax[i + 1].text(s=s, fontsize=25, x=0.5, y=-0.15, horizontalalignment='center',
                                   transform=ax[i + 1].transAxes)
                    ax[i + 1].set_xticks([])
                    ax[i + 1].set_yticks([])
                    ax[i + 1].imshow(denoise_result, "gray", vmin=(-160 + 1024) / 4096, vmax=(240 + 1024) / 4096)
            if not only_metric:
                plt.savefig(save_path + "/deImg.png", dpi=200)

        # Note: mode="progressive",show results from "progressive_denoiser" and "fdct" and "ldct"
        if mode == "progressive":
            self.metric_calculate(mode="LDCT", it=0, denoise_result=self.ldct_np)
            if not only_metric:
                fig, ax = plt.subplots(2, 1 + max(len(self.progressive_denoise_result),
                                                  len(self.proj_denoise_convert2img_result)),
                                       figsize=(7 * (1 + max(len(self.progressive_denoise_result),
                                                             len(self.proj_denoise_convert2img_result))), 16))
                # LDCT
                ax[0, 0].set_title("LDCT", fontsize=35, y=1.02)
                s = 'PSNR={:.2f}'.format(self.metric_instance.LDCT.psnr_iter_0) + ' , ' + 'SSIM={:.2f}'.format(
                    self.metric_instance.LDCT.ssim_iter_0)
                ax[0, 0].text(x=0.5, y=-0.09, s=s, fontsize=25, horizontalalignment='center',
                              transform=ax[0, 0].transAxes)
                ax[0, 0].set_xticks([])
                ax[0, 0].set_yticks([])
                ax[0, 0].imshow(self.ldct_np, "gray", vmin=(-160 + 1024) / 4096, vmax=(240 + 1024) / 4096)

            for i in range(1, len(self.proj_denoise_convert2img_result) + 1):
                denoise_result = miu2pixel(self.proj_denoise_convert2img_result[f"iter_{i}"][0, 0])
                self.metric_calculate(mode="deProj", it=i, denoise_result=denoise_result)
                if not only_metric:
                    ax[0, i].set_title("Proj iter{}".format(i), fontsize=35, y=1.02)
                    s = 'PSNR={:.2f}'.format(
                        self.metric_instance.deProj[f"psnr_iter_{i}"]) + ' , ' + 'SSIM={:.2f}'.format(
                        self.metric_instance.deProj[f"ssim_iter_{i}"])
                    ax[0, i].text(s=s, fontsize=25, x=0.5, y=-0.09, horizontalalignment='center',
                                  transform=ax[0, i].transAxes)
                    ax[0, i].set_xticks([])
                    ax[0, i].set_yticks([])
                    ax[0, i].imshow(denoise_result, "gray", vmin=(-160 + 1024) / 4096, vmax=(240 + 1024) / 4096)

            img_its = len(self.progressive_denoise_result)
            for i in range(1, img_its + 1):
                r_it = img_its + 1 - i
                denoise_result = miu2pixel(self.progressive_denoise_result[f"iter_{r_it}"][0, 0])
                self.metric_calculate(mode="deProg", it=r_it, denoise_result=denoise_result)
                if not only_metric:
                    s = 'PSNR={:.2f}'.format(
                        self.metric_instance.deProg[f"psnr_iter_{r_it}"]) + ' , ' + 'SSIM={:.2f}'.format(
                        self.metric_instance.deProg[f"ssim_iter_{r_it}"])
                    ax[1, i].set_title("Img iter{}".format(r_it), fontsize=35, y=1.02)
                    ax[1, i].text(s=s, fontsize=25, x=0.5, y=-0.09, horizontalalignment='center',
                                  transform=ax[1, i].transAxes)
                    ax[1, i].set_xticks([])
                    ax[1, i].set_yticks([])
                    ax[1, i].imshow(denoise_result, "gray", vmin=(-160 + 1024) / 4096, vmax=(240 + 1024) / 4096)
            if not only_metric:
                ax[1, 0].set_title("FDCT", fontsize=35, y=1.02)
                ax[1, 0].set_xticks([])
                ax[1, 0].set_yticks([])
                ax[1, 0].imshow(self.fdct, "gray", vmin=(-160 + 1024) / 4096, vmax=(240 + 1024) / 4096)
                plt.savefig(save_path + "/progressive.png", dpi=100)
        if not display and fig is not None:
            plt.close(fig)

    def result_data_save(self, data_save=True):
        save_path = self.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if data_save:
            for ftype, fdata in zip(
                    ["prog_denoise_result", "proj_denoise_result", "img_denoise_result", "proj_denoise_result_2img"],
                    [self.progressive_denoise_result, self.proj_denoise_result, self.img_denoise_result,
                     self.proj_denoise_convert2img_result]):
                if len(fdata) > 0:
                    np.savez_compressed(save_path + f'/{ftype}.npz', **fdata)

        # save metric
        info_json = json.dumps(self.metric_instance, sort_keys=False, indent=4, separators=(',', ': '))
        f = open(save_path + '/metric.json', 'w')
        f.write(info_json)
        f.close()
        pass

    def save_path_load(self, epoch, patient_name, slice_name):
        self.save_path = self.save_root_path + f'/Save_Iter_{epoch}' + f'/{patient_name}' + f'/{slice_name}'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def metric_calculate(self, mode="LDCT", **kwargs):
        i = kwargs["it"]
        ld = kwargs["denoise_result"]
        ld[np.isnan(ld)] = 0.5
        if 'psnr' in self.opt.metrics:
            self.metric_instance[mode]["psnr_iter_{}".format(i)] = compare_psnr(self.fdct, ld,
                                                                                data_range=1)
        if 'ssim' in self.opt.metrics:
            self.metric_instance[mode]["ssim_iter_{}".format(i)] = compare_ssim(self.fdct, ld,
                                                                                win_size=11, data_range=1)
        fd = torch.from_numpy(self.fdct)[None, None]
        ld = torch.from_numpy(ld[None, None])
        if 'fsim' in self.opt.metrics:
            self.metric_instance[mode]["fsim_iter_{}".format(i)] = fsim(fd, ld, data_range=1, chromatic=False).item()
        if 'vif' in self.opt.metrics:
            self.metric_instance[mode]["vif_iter_{}".format(i)] = vif_p(fd, ld, data_range=1).item()
        if 'nqm' in self.opt.metrics:
            self.metric_instance[mode]["nqm_iter_{}".format(i)] = NQM(self.fdct, kwargs["denoise_result"]).item()

    def metric_update(self):
        self.metric_each_sample.append(self.metric_instance)

    def metric_total_save(self, epoch):
        d = DotDict()
        self.metric_total = DotDict()
        metric_mean = DotDict()
        for m in self.metric_each_sample:
            dict_add(metric_mean, m, d)
        dict_mean(metric_mean, d)
        d = DotDict()
        for m in self.metric_each_sample:
            dict_value_minus_mean_square(m, metric_mean, d)
        dict_std(metric_mean, d)
        self.metric_total = metric_mean
        print(self.metric_total)
        # save metric
        info_json = json.dumps(self.metric_total, sort_keys=False, indent=4, separators=(',', ': '))
        f = open(self.save_root_path + f'/Save_Iter_{epoch}/metric.json', 'w')
        f.write(info_json)
        f.close()


def weight_lambda(x, f1, f2):
    if x < 1:
        return f1(1)
    elif 1 <= x <= 1.7:
        return f1(x)
    elif 1.7 < x <= 2.75:
        return f2(x)
    elif x > 2.75:
        return f2(2.75)


def curve_init():
    # %%
    x = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
    y = [20, 17.5, 15, 12, 8.5, 5, 2, 1]
    x_ = [1.7, 1.8, 2.0, 2.2, 2.35, 2.5, 3]
    y_ = [1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05]
    z1 = np.polyfit(x, y, 4)
    p1 = np.poly1d(z1)
    z2 = np.polyfit(x_, y_, 2)
    p2 = np.poly1d(z2)
    return functools.partial(np.vectorize(weight_lambda, otypes=[np.float32], excluded=['f1', 'f2']), f1=p1, f2=p2)


def proj_curv_init():
    # %%
    x = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
    y = [20, 17.5, 15, 12, 8.5, 7.5, 5, 4]
    x_ = [1.7, 1.8, 2.0, 2.2, 2.35, 2.5, 3, 3.5]
    y_ = [4, 3, 2, 1, 0.5, 0.3, 0.1, 0.01]
    z1 = np.polyfit(x, y, 4)
    p1 = np.poly1d(z1)
    z2 = np.polyfit(x_, y_, 2)
    p2 = np.poly1d(z2)
    return functools.partial(np.vectorize(weight_lambda, otypes=[np.float32], excluded=['f1', 'f2']), f1=p1, f2=p2)


def tensor_sharpen(img_in, N=60):
    if N != -1:
        B = img_in.shape[0]
        shapen_filter = torch.tensor([[-2, -2, -2],
                                      [-2, N, -2],
                                      [-2, -2, -2]])[None, None, :, :].float().repeat(B, 1, 1, 1).to(img_in.device) / (
                                N - 16)
        img_arr = torch.nn.functional.conv2d(img_in, shapen_filter, stride=1, padding=1)
    else:
        img_arr = img_in
    return img_arr
