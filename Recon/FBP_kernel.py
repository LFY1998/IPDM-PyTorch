import numba
import numpy as np
import torch
from matplotlib import pyplot as plt
from numba import jit, cuda, prange
import math

numba.config.NUMBA_DEFAULT_NUM_THREADS = 8


class Grid:
    def __init__(self):
        self.yg = None
        self.xg = None
        self.N = None
        self.L = None

    def getgrid(self, v1, v2):
        N_ = v1
        self.L = v2
        # disp(strcat('图像网格长度:', num2str(2 * grid.L), 'cm'));
        self.N = N_
        self.xg = self.L * np.linspace(-1, 1, N_ + 1)
        self.yg = self.L * np.linspace(-1, 1, N_ + 1)


class FBP:
    def __init__(self, device="cpu"):
        self.device = device
        # 读入数据 行角度  列探测器方向
        # 配置重建参数表
        self.os_ = 59.5
        self.od = 108.56 - 59.5
        self.M = 912
        self.T = 0.0010125
        self.N = 512
        self.da = self.T
        self.theta = np.arange(0, 359.82 + 0.18, 0.18) / 180 * np.pi
        self.nda = np.arange((-self.M / 2 + 0.5 + 3.75) * self.da, (self.M / 2 - 0.5 + 3.75 + 1) * self.da,
                             self.da).astype("float32")
        self.sp0 = [-self.os_, 0]
        self.dp0 = [(self.os_ + self.od) * np.cos(self.nda) - self.os_, (self.os_ + self.od) * np.sin(self.nda)]
        self.grid = Grid()
        self.grid.getgrid(self.N, 21)
        # 计算探测器角度分布

        # % 1.修正投影函数
        self.D = np.abs(self.sp0[0])
        self.N = self.dp0[0].size  # 探测器个数
        self.M = self.theta.size  # 角度数
        # 卷积核
        h_RL = np.zeros((2 * self.N - 1, 1))
        ngarma = np.arange(-self.N + 1, self.N, 2) * self.da
        h_RL[0:2 * self.N - 1:2] = (-0.5 / np.pi ** 2. / (np.sin(ngarma) ** 2))[:, None]
        h_RL[self.N - 1] = 1 / 8 / self.da ** 2
        self.h_RL = (h_RL * self.da).astype("float32")
        # % 预先计算所有点的极坐标
        self.r, self.phi = self.getrphi(np.arange(0, self.grid.N ** 2))
        self.r.resize(self.grid.N, self.grid.N)
        self.phi.resize(self.grid.N, self.grid.N)

        if device != "cpu":
            cuda.select_device(int(device.split(":")[-1]))
            self.theta_gpu = cuda.to_device(self.theta.astype(np.float32))
            self.h_RL_gpu = cuda.to_device(self.h_RL.astype(np.float32))
            self.r_gpu = cuda.to_device(self.r.astype(np.float32))
            self.phi_gpu = cuda.to_device(self.phi.astype(np.float32))

    def getrphi(self, isect):
        #  %获取当前像素点的极坐标r phi
        # %例如
        cx = self.grid.N / 2
        cy = self.grid.N / 2
        i, j = np.unravel_index(isect, (self.grid.N, self.grid.N))
        i += 1
        j += 1
        y = (self.grid.N + 1 - i - cx - 0.5) * 2 * self.grid.L / self.grid.N
        x = (j - cy - 0.5) * 2 * self.grid.L / self.grid.N
        r = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan(y / x)
        # %归入0~2pi的范畴
        phi[x < 0] = phi[x < 0] + np.pi
        phi[phi < 0] = phi[phi < 0] + 2 * np.pi
        return r, phi

    def convert(self, pj, flip=True):
        """
        :param flip: flip the projection raw
        :param pj: projection data, ndarray[mini-batch,H,W]/Tensor
        :return: image:ndarray
        """
        dtype = "array"
        if isinstance(pj, torch.Tensor):
            pj = pj.detach().cpu().numpy()
            dtype = "tensor"

        if pj.shape == (self.M, self.N):
            pj = pj[None]
        if flip:
            pj = np.flip(pj, 2).astype("float32")
        BS = pj.shape[0]
        I_ = np.zeros((BS, self.grid.N, self.grid.N)).astype(np.float32)

        pj = pj * (self.D * np.cos(self.nda))[None, :].repeat(self.M, 0)
        pj = (pj * (self.theta[1] - self.theta[0])).astype("float32")
        pj_out = np.zeros_like(pj).astype("float32")

        if self.device == "cpu":
            pj_out = conv_pj(pj_out, pj, self.h_RL, self.M, self.N, BS)
            I_ = fbp_cpu(I_, BS, pj_out, self.phi, self.r, self.D, self.grid.N, self.M, self.N, self.theta, self.da,
                         self.nda)
        else:
            pj_out_gpu = cuda.to_device(pj_out)
            conv_kernel[(32, 32, 128), (8, 8, 8)](pj_out_gpu, pj, self.h_RL_gpu, self.M, self.N, BS)
            fbp_kernel[(64, 64, 1), (8, 8, BS)](I_, BS, pj_out_gpu, self.phi_gpu, self.r_gpu, self.D, self.grid.N,
                                                self.M, self.N, self.theta_gpu, self.da, self.nda)
        if flip:
            I_ = np.flip(I_, 2)
        if dtype == "tensor":
            return torch.from_numpy(I_.copy())
        else:
            return I_.copy()


@jit(nopython=True, parallel=True)
def conv_pj(pj_out, pj_raw, kernel, M, N, BS):
    for k in prange(BS):
        for t in prange(M):
            convres = np.convolve(kernel[:, 0], pj_raw[k, t, :])
            pj_out[k, t, :] = convres[N - 1: 2 * N - 1]
    return pj_out


@cuda.jit
def conv_kernel(pj_out, pj_raw, kernel, M, N, BS):
    ix, iy, it = cuda.grid(3)
    threads_per_grid_x, threads_per_grid_y, threads_per_grid_t = cuda.gridsize(3)
    for k in range(BS):
        for t in range(it, M, threads_per_grid_t):
            for i in range(N - 1 + iy, 2 * N - 1, threads_per_grid_y):
                for j in range(ix, i + 1, threads_per_grid_x):
                    if j <= N - 1:
                        cuda.atomic.add(pj_out, (k, t, i - (N - 1)), pj_raw[k, t, j] * kernel[i - j, 0])


@cuda.jit
def fbp_kernel(I, BS, pj, phi, r, D, gridN, M, N, theta, da, nda):
    ix, iy, it = cuda.grid(3)
    threads_per_grid_x, threads_per_grid_y, threads_per_grid_t = cuda.gridsize(3)
    for k in range(it, BS, threads_per_grid_t):
        for t in range(0, M, 1):
            beta = theta[t] - np.pi / 2  # % 投影时的theta和平行束一样是和x轴的夹角，公式上的beta是和y轴的夹角
            for i in range(iy, gridN, threads_per_grid_y):
                for j in range(ix, gridN, threads_per_grid_x):
                    # % 穿过该像素的射线与探测器的交点
                    th = np.pi / 2 + beta + phi[i, j]
                    alpha = math.atan(r[i, j] * math.sin(th) / (D + r[i, j] * math.cos(th)))
                    curdet = math.floor((alpha - nda[0]) / da + 0.5)
                    if 0 < curdet < N:
                        lam = (alpha - nda[0]) / da + 0.5 - curdet
                        L = r[i, j] * math.sin(th) / math.sin(alpha)
                        I[k, i, j] = I[k, i, j] + (
                                    (1 - lam) * pj[k, t, int(curdet - 1)] + lam * pj[k, t, int(curdet)]) / L ** 2


@jit(nopython=True, parallel=True)
def fbp_cpu(I, BS, pj, phi, r, D, gridN, M, N, theta, da, nda):
    # % 3.
    # 卷积加权反投影
    for t in prange(0, M):
        beta = theta[t] - np.pi / 2  # % 投影时的theta和平行束一样是和x轴的夹角，公式上的beta是和y轴的夹角
        for k in prange(BS):
            for i in prange(gridN):
                for j in prange(gridN):
                    # % 穿过该像素的射线与探测器的交点
                    th = np.pi / 2 + beta + phi[i, j]
                    alpha = np.arctan(r[i, j] * np.sin(th) / (D + r[i, j] * np.cos(th)))
                    curdet = np.floor((alpha - nda[0]) / da + 0.5)
                    if 0 < curdet < N:
                        lam = (alpha - nda[0]) / da + 0.5 - curdet
                        L = r[i, j] * np.sin(th) / np.sin(alpha)
                        I[k, i, j] = I[k, i, j] + (
                                    (1 - lam) * pj[k, t, int(curdet - 1)] + lam * pj[k, t, int(curdet)]) / L ** 2
    return I


if __name__ == '__main__':
    # device="cuda:0,1..."/"cpu"
    device = "cuda:0"
    fbp = FBP(device=device)
    proj_path = r"/media/ubuntu/Elements/siemens/train/train_proj_txt/P00578242/AAAAAAHT.txt"
    pj_raw = np.fromfile(proj_path, "float32")
    pj_raw.resize(2000, 912)
    I = fbp.convert(pj_raw.astype("float32"))
    plt.imshow(I[0, :, :], "gray")
    plt.show()
