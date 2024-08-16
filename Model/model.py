from abc import abstractmethod
from copy import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import cuda
from sklearn.preprocessing import PowerTransformer
from Dataset.npz_data_loader import miu2pixel


# %%
# use sinusoidal position embedding to encode time step (https://arxiv.org/abs/1706.03762)
def timestep_embedding(timesteps, dim, max_period=10000, dtype=torch.float32):
    """
    Create sinusoidal timestep embeddings.
    :param dtype:
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half) / half
    ).type(dtype).to(device=timesteps.device)
    args = timesteps[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# %%
# define TimestepEmbedSequential to support `time_emb` as extra input
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, size):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, Upsample):
                x = layer(x, size)
            else:
                x = layer(x)
        return x


import math


def factor(num):
    factors = []
    for_times = int(math.sqrt(num))
    for i in range(for_times + 1)[1:]:
        if num % i == 0:
            factors.append(i)
            t = int(num / i)
            if not t == i:
                factors.append(t)
    return np.array(factors)


# use GN for norm layer
def norm_layer(channels):
    if channels % 32 == 0:
        return nn.GroupNorm(32, channels)
    elif channels < 32:
        return nn.GroupNorm(channels, channels)
    else:
        f = factor(channels)
        idx = np.argmin((f - 32) ** 2)
        return nn.GroupNorm(f[idx], channels)


# %%
# Residual block
class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        # pojection for time step embedding
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        """
        h = self.conv1(x)
        # Add time step embeddings
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


# %%
# Attention block with shortcut
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B * self.num_heads, -1, H * W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x


# %%
# upsample
class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


# downsample
class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(stride=2, kernel_size=2)

    def forward(self, x):
        return self.op(x)


# %%
# The full UNet model with attention and timestep embedding
class UNetModel(nn.Module):
    def __init__(
            self,
            in_channels=3,
            model_channels=128,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0,
            channel_mult=(1, 2, 2, 2),
            conv_resample=True,
            num_heads=4,
            pre_downsample_times=1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads

        # time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # down blocks
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                nn.Conv2d(in_channels, int(channel_mult[0] * model_channels), kernel_size=3, padding=1))
        ])

        ch = int(channel_mult[0] * model_channels)
        down_block_chans = [ch]
        ds = 1
        channel_mult = channel_mult[1:]
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(ch, int(mult * model_channels), time_embed_dim, dropout)
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1:  # don't use downsample for the last stage
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2

        # middle block
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResidualBlock(ch, ch, time_embed_dim, dropout)
        )

        # up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(
                        ch + down_block_chans.pop(),
                        int(model_channels * mult),
                        time_embed_dim,
                        dropout
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))
            pass

        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x H x W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        # time step embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels, dtype=x.dtype))

        # down stage
        h = x
        # for md in self.pd:

        for module in self.down_blocks:
            h = module(h, emb, None)
            hs.append(h)
        # middle stage
        h = self.middle_block(h, emb, None)
        # up stage
        h_ = hs.pop()
        for module in self.up_blocks:
            cat_in = torch.cat([h, h_], dim=1)
            if len(hs) != 0:
                h_ = hs.pop()
            h = module(cat_in, emb, (h_.shape[-2], h_.shape[-1]))
        return self.out(h)


# %%
# beta schedule
def linear_beta_schedule(timesteps, schedule_power):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return (torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)) ** schedule_power


def sigmoid_beta_schedule(timesteps, schedule_power):
    steps = timesteps + 1
    x = (torch.sigmoid(torch.linspace(-steps / schedule_power, steps / schedule_power, steps, dtype=torch.float64)))
    return x


@cuda.jit
def condition_lambda_ratio_cuda(I, idx, B, H, W, timesteps, lambda_):
    """
    对[B,H,W]大小的图像每个像素计算条件衰减值
    @param I:
    @param idx:
    @param B:
    @param H:
    @param W:
    @param timesteps:
    @param lambda_:
    """
    s = 0.008
    ix, iy, it = cuda.grid(3)
    threads_per_grid_x, threads_per_grid_y, threads_per_grid_t = cuda.gridsize(3)
    for k in range(it, B, threads_per_grid_t):
        for i in range(iy, H, threads_per_grid_y):
            for j in range(ix, W, threads_per_grid_x):
                a0 = (math.cos(((idx[0] / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2) ** lambda_[k, 0, i, j]
                a1 = (math.cos(((idx[1] / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2) ** lambda_[k, 0, i, j]
                a2 = (math.cos(((idx[2] / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2) ** lambda_[k, 0, i, j]
                a1 = a1 / a0
                a2 = a2 / a0
                I[k, 0, i, j] = 1 - (a2 / a1)


def condition_lambda_ratio(idx, timesteps, s=0.008, lambda_=1.):
    """
    cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672
    """
    x = np.array([0, idx, idx + 1])
    alphas_cumprod = (np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2) ** lambda_
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[2] / alphas_cumprod[1])
    return np.clip(betas, 0.3, 0.999)


def cosine_beta_schedule(timesteps, s=0.008, schedule_power=1):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = (torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2) ** schedule_power
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# %%
class GaussianDiffusion:
    def __init__(
            self,
            timesteps=1000,
            beta_schedule='linear',
            schedule_power=1
    ):
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps, schedule_power=schedule_power)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps, schedule_power=schedule_power)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas

        self.alphas = 1. - self.betas
        # 累乘，输出每一个位置的结果（该位置之前元素的累乘）
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # self.posterior_variance = self.betas
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1 = (
                self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * torch.sqrt(self.alphas)
                / (1.0 - self.alphas_cumprod)
        )

    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    def lambda_t_calculate(self, eta=0.9):
        lambda_t = torch.abs(
            (1 - eta + eta * self.alphas - self.alphas_cumprod) * torch.sqrt(self.alphas_cumprod_prev) / (
                    1 - self.alphas_cumprod))
        lambda_cumprod = torch.cumprod(lambda_t, axis=0)
        return lambda_cumprod

    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_sample_inverse(self, x_t, x_start, t):
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return (x_t - sqrt_alphas_cumprod_t * x_start) / sqrt_one_minus_alphas_cumprod_t
        # return x_t-sqrt_alphas_cumprod_t*x_start

    # Get the mean and variance of q(x_t | x_0).
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # compute x_0 from x_t and pred noise: the reverse of `q_sample`
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, x_t, t, clip_denoised=False):
        # predict noise using model
        pred_noise = model(x_t, t)
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    def std(self, data: torch.Tensor):
        return (data - data.mean()) / torch.std(data)

    def p_mean_variance_condition(self, model, x_t, x_0, t, lambda_, clip_denoised=False, mode="proj"):
        # predict noise using model
        pred_noise = model(x_t, t)
        condition_noise = self.q_sample_inverse(x_t, x_0, t).type(pred_noise.dtype)
        pred_noise = self.std((1 - lambda_) * self.std(pred_noise) + lambda_ * self.std(condition_noise))
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1, max=1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_condition(self, model, x_t, x_0, t, clip_denoised=True, lambda_=1., mode="img"):
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance_condition(model, x_t, x_0, t, lambda_,
                                                                           clip_denoised=clip_denoised, mode=mode)
        noise = torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = ((t != 0).type(x_t.dtype).view(-1, *([1] * (len(x_t.shape) - 1))))
        # nonzero_mask =1
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img

    @torch.no_grad()
    def guided_reverse_process(self, model, img, t_start=None, clip=True,
                               lambda_ratio=1, eta=0.5,
                               save_states=False, mode="img", constant_guidance=None, **kwargs):
        if kwargs["only_convertor"]:
            # 用于测试FBP/TV/ART指标
            return [img], None, None
        # 条件图片与denoised图片线性组合输入，组合权重应当满足前期贴近img_condition，后期贴近denoised_img
        # 加入加噪的原图，用于矫正去噪过程
        device = next(model.parameters()).device
        bs = img.shape[0]
        img_with_noise = img.clone()
        # start from pure noise (for each example in the batch)
        img_iters = []
        adaptive = False
        if t_start is not None:
            t_start_list = copy(t_start)
        else:
            t_start_list = [20]
            adaptive = True
        img_reverse = []
        imgs = img.clone()
        noise_strength = None
        iters = 0
        delt = None
        l_s = None
        while t_start_list:
            ts = t_start_list.pop(0)
            img_with_noise = self.q_sample(img_with_noise, (torch.tensor([ts])).to(device).repeat(bs))
            lambda_schedule_cosine = cosine_beta_schedule(ts, schedule_power=lambda_ratio).to(device)  # 0->1
            lambda_schedule_condition = condition_lambda_ratio_cuda
            # with tqdm(total=ts, desc='sampling loop time step', position=0, leave=False) as T:
            for i in reversed(range(0, ts)):
                if constant_guidance is None:
                    if iters == 0:
                        l_s = lambda_schedule_cosine[i]
                    elif iters >= 1:
                        I = np.zeros_like(delt)
                        lambda_schedule_condition[(64, 64, 1), (8, 8, bs)](I, np.array([0, i, i + 1]), bs,
                                                                           delt.shape[-2],
                                                                           delt.shape[-1], ts, delt)
                        I = np.clip(I, 0.05, 0.99)
                        l_s = torch.nn.functional.interpolate(
                            torch.from_numpy(I).to(device), size=(img.shape[-2], img.shape[-1]), mode="nearest")
                else:
                    l_s = constant_guidance
                img_with_noise = self.p_sample_condition(model, img_with_noise.type(img.dtype), imgs,
                                                         torch.full((1,), i, device=device, dtype=torch.long),
                                                         clip_denoised=clip, lambda_=l_s, mode=mode)
                # T.update(1)
                if save_states:
                    img_reverse.append(img_with_noise.detach().cpu().numpy())
            if clip:
                if mode == "img":
                    img_with_noise = img_with_noise.clamp(0, 1)
                else:
                    img_with_noise = img_with_noise.clamp(min=0)
            if iters == 0 and constant_guidance is None:
                if mode == "img":
                    delt = torch.abs(miu2pixel(img_with_noise) - miu2pixel(img.clone()))
                    delt = torch.nn.functional.avg_pool2d(delt, kwargs["kernel_size_img"])
                    delt = delt - torch.median(delt)
                    delt[delt <= 0] = 0
                    delt = kwargs["lambda_curve"](torch.exp(kwargs["amplitude_img"] * delt).cpu().numpy())
                    # Section:根据delt设定迭代次数
                    if adaptive:
                        if kwargs["noise_strength"] == "high":
                            t_start_list = [15, 15, 15]
                            eta = 0.6
                            l_s = 0.4
                        elif kwargs["noise_strength"] == "mid":
                            t_start_list = [15, 12, 10]
                            eta = 0.55
                            l_s = 0.45
                        elif kwargs["noise_strength"] == "low" or kwargs["noise_strength"] is None:
                            t_start_list = [10, 10, 10]
                            eta = 0.5
                            l_s = 0.5
                elif mode == "proj":
                    delt = torch.abs(img_with_noise - img)
                    delt = delt - torch.median(delt)
                    delt = torch.nn.functional.avg_pool2d(delt, kwargs["kernel_size_proj"])
                    delt[delt <= 0] = 0
                    delt = torch.exp(kwargs["amplitude_proj"] * delt).cpu().numpy()
                    if adaptive:
                        if delt.max() >= 30:
                            t_start_list = [30, 25, 20]
                            noise_strength = "high"
                            eta = 0.6
                        elif delt.max() >= 4.5:
                            t_start_list = [20, 18, 15]
                            noise_strength = "mid"
                            eta = 0.5
                        else:
                            t_start_list = [15, 15, 15]
                            noise_strength = "low"
                            eta = 0.5
                    delt = kwargs["lambda_curve"](delt)
            # 每次iter，img应该更新为上次去噪的结果
            if kwargs["normal"]:
                img_iters.append(yeo_johnson_inverse_transform(img_with_noise.contiguous(), kwargs["transformer"]))
            else:
                img_iters.append(img_with_noise.contiguous())
            # 更新引导
            # 如果固定条件引导，则第一次迭代后更换，若为自适应，则第一次不更换
            if constant_guidance is None:
                # 自适应
                if iters >= 1:
                    if mode == "proj":
                        imgs = eta * img_with_noise.clone() + (1 - eta) * img
                    if mode == "img":
                        imgs = eta * img_with_noise.clone() + (0.95 - eta) * img + 0.05 * kwargs["ldct"]
                if iters == 0:
                    img_with_noise = img.clone()
            else:
                if mode == "proj":
                    imgs = eta * img_with_noise.clone() + (1 - eta) * img
                if mode == "img":
                    imgs = eta * img_with_noise.clone() + (0.95 - eta) * img + 0.05 * kwargs["ldct"]
            iters += 1
        if len(img_iters) > 1:
            img_iters.append((img_iters[-1] + img_iters[-2]) / 2)
        if adaptive:
            return img_iters[1:], img_reverse, noise_strength
        else:
            return img_iters, img_reverse, noise_strength

    # compute train losses
    def train_losses(self, model, x_start, t):
        # generate random noise
        noise = torch.randn_like(x_start)
        # get x_t
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = model(x_noisy, t)
        loss = F.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def ddim_sample(
            self,
            sample_img,
            model,
            condition,
            t_start,
            condition_lambda=0.5,
            batch_size=1,
            ddim_timesteps=2,
            ddim_discr_method="uniform",
            ddim_eta=0.0,
            clip_denoised=True):
        # make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = t_start // ddim_timesteps
            # ddim_timestep_seq = np.asarray(list(range(self.timesteps-1, 0,-c)))
            ddim_timestep_seq = np.linspace(t_start - 1, 0, ddim_timesteps + 1).astype(int)[0:-1]
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                    (np.linspace(0, np.sqrt(self.timesteps * .8), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        # ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(ddim_timestep_seq[1:], np.array([0]))

        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)

        # with tqdm(total=ddim_timesteps, position=0) as T:
        for i in range(ddim_timesteps):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)

            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, sample_img.shape)

            # 2. predict noise using model
            pred_noise = model(sample_img, t)
            condition_noise = self.q_sample_inverse(sample_img, condition, t).type(pred_noise.dtype)
            pred_noise = self.std(
                (1 - condition_lambda) * self.std(pred_noise) + condition_lambda * self.std(condition_noise))
            # 3. get the predicted x_0
            pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)

            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            # sigmas_t = ddim_eta * self._extract(self.posterior_variance,t, sample_img.shape)

            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) * pred_noise
            sigmas_t = ddim_eta * self._extract(self.posterior_variance, t, sample_img.shape)
            # 6. compute x_{t-1} of formula (12)
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(
                sample_img)

            sample_img = x_prev
            # T.set_description(desc="sampling loop time step")
            # T.update(1)
        # sample_img = self.p_sample(model, sample_img, torch.full((batch_size,), 0, device=device, dtype=torch.long),
        #                            clip_denoised=clip_denoised)

        return sample_img

    @torch.no_grad()
    def sparse_guided_reverse_process(self, model,
                                      condition,
                                      t_start,
                                      condition_lambda_max=0.5,
                                      condition_lambda_min=0.25,
                                      batch_size=1,
                                      ddim_timesteps=[2],
                                      ddim_discr_method="uniform",
                                      ddim_eta=0.0,
                                      eta=0.5,
                                      clip_denoised=True):
        device = next(model.parameters()).device
        sample_img = self.q_sample(condition, (torch.tensor([t_start[0]])).to(device).repeat(batch_size))
        condition_ = condition.clone()
        iteration_num = len(t_start)
        step = (condition_lambda_max - condition_lambda_min) / iteration_num
        condition_lambda = np.arange(condition_lambda_max, condition_lambda_min - step, -step)
        result = []
        for i, t in enumerate(t_start):
            sample_img = self.ddim_sample(
                sample_img=sample_img,
                model=model,
                condition=condition,
                t_start=t,
                condition_lambda=condition_lambda[i],
                batch_size=batch_size,
                ddim_timesteps=ddim_timesteps[i],
                ddim_discr_method=ddim_discr_method,
                ddim_eta=ddim_eta,
                clip_denoised=clip_denoised)
            condition = eta * sample_img.clone() + (1 - eta) * condition_
            result.append(sample_img.clone())
        return result


def yeo_johnson_transform(img_tensor):
    """
    对输入的图像数据进行 Yeo-Johnson 变换。

    Args:
    - img_tensor (torch.Tensor): 输入的图像数据张量，大小为 [B, 1, H, W]。

    Returns:
    - transformed_img_tensor (torch.Tensor): 变换后的图像数据张量，大小与输入相同。
    - transformer (sklearn.preprocessing.PowerTransformer): 存储变换所使用的 transformer，用于逆变换。
    """
    # 将图像数据转换为 numpy 数组并展平
    img_np = img_tensor.cpu().numpy().reshape(-1, 1)

    # 使用 Yeo-Johnson 变换器进行变换
    transformer = PowerTransformer(method='yeo-johnson')
    transformed_img_np = transformer.fit_transform(img_np)

    # 将变换后的数据重新转换为 Tensor 并保持原始形状
    transformed_img_tensor = torch.from_numpy(transformed_img_np.reshape(img_tensor.shape)).to(img_tensor.device)

    return transformed_img_tensor, transformer


def yeo_johnson_inverse_transform(transformed_img_tensor, transformer):
    """
    对经过 Yeo-Johnson 变换后的图像数据进行逆变换。

    Args:
    - transformed_img_tensor (torch.Tensor): 经过 Yeo-Johnson 变换后的图像数据张量，大小与原始数据相同。
    - transformer (sklearn.preprocessing.PowerTransformer): 存储用于变换的 transformer 对象。

    Returns:
    - original_img_tensor (torch.Tensor): 逆变换后的原始图像数据张量，大小与输入相同。
    """
    # 将 Tensor 转换为 numpy 数组并展平
    transformed_img_np = transformed_img_tensor.cpu().numpy().reshape(-1, 1)

    # 使用 transformer 对象进行逆变换
    original_img_np = transformer.inverse_transform(transformed_img_np)

    # 将逆变换后的数据重新转换为 Tensor 并保持原始形状和设备位置
    original_img_tensor = torch.from_numpy(original_img_np.reshape(transformed_img_tensor.shape)).to(
        transformed_img_tensor.device)

    return original_img_tensor
