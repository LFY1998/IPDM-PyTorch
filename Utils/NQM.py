import numpy as np
from numpy import cos, log2, pi, real, log10
from numpy.fft import fft2, fftshift, ifft2


def ctf(f_r):
    if isinstance(f_r, float) or isinstance(f_r, int):
        y = 1. / (200 * (2.6 * (0.0192 + 0.114 * (f_r)) * np.exp(-(0.114 * (f_r)) ** 1.1)))
        return y
    s = np.shape(f_r)
    f_r = f_r.flatten()
    y = 1. / (200 * (2.6 * (0.0192 + 0.114 * (f_r)) * np.exp(-(0.114 * (f_r)) ** 1.1)))
    return y.reshape(np.shape(f_r))


def find(condition):
    return np.nonzero(condition)


def cmaskn_modified(c, ci, a, ai, i):
    H, W = np.shape(c)
    ci = ci.flatten()
    c = c.flatten()
    t = find(np.abs(ci) > 1)
    ci[t] = 1
    ai = ai.flatten()
    a = a.flatten()
    ct = ctf(i)
    T = ct * (.86 * ((c / ct) - 1) + .3)

    a1 = find((np.abs(ci - c) - T) < 0)

    ai[a1] = a[a1]

    return ai.reshape(H, W)


def gthresh_modified(x, T, z):
    H, W = np.shape(x)
    x = x.flatten()
    z = z.flatten()
    a = find(np.abs(x) < T)
    z[a] = np.zeros(np.shape(a))
    return z.reshape(H, W)


def NQM(image_origin, image_query, view_angle=1, ):
    """
    :param image_origin: 2-d numpy array
    :param image_query: 2-d numpy array
    :param view_angle: I don't know what this parameter means, I'm just ruthless matlab code translation machine.
                        The god knows how much the default value is, may be 1, who knows ?
    :return:
    """
    i = complex(0, 1)
    O = image_origin
    I = image_query
    VA = view_angle
    x, y = np.shape(O)
    xplane, yplane = np.meshgrid(np.arange(-y / 2, y / 2), np.arange(-x / 2, x / 2))
    plane = (xplane + i * yplane)
    r = np.abs(plane)
    FO = fft2(O)
    FI = fft2(I)
    G_0 = 0.5 * (
            1 + cos(pi * log2((r + 2) * ((r + 2 <= 4) * (r + 2 >= 1)) + 4 * (~((r + 2 <= 4) * (r + 2 >= 1)))) - pi))
    G_1 = 0.5 * (1 + cos(pi * log2(r * ((r <= 4) * (r >= 1)) + 4 * (~((r <= 4) * (r >= 1)))) - pi))
    G_2 = 0.5 * (1 + cos(pi * log2(r * ((r >= 2) * (r <= 8)) + .5 * (~((r >= 2) * (r <= 8))))))
    G_3 = 0.5 * (1 + cos(pi * log2(r * ((r >= 4) * (r <= 16)) + 4 * (~((r >= 4) * (r <= 16)))) - pi))
    G_4 = 0.5 * (1 + cos(pi * log2(r * ((r >= 8) * (r <= 32)) + .5 * (~((r >= 8) * (r <= 32))))))
    G_5 = 0.5 * (1 + cos(pi * log2(r * ((r >= 16) * (r <= 64)) + 4 * (~((r >= 16) * (r <= 64)))) - pi))
    GS_0 = fftshift(G_0)
    GS_1 = fftshift(G_1)
    GS_2 = fftshift(G_2)
    GS_3 = fftshift(G_3)
    GS_4 = fftshift(G_4)
    GS_5 = fftshift(G_5)

    L_0 = ((GS_0) * FO)
    LI_0 = (GS_0 * FI)

    l_0 = real(ifft2(L_0))
    li_0 = real(ifft2(LI_0))

    A_1 = GS_1 * FO
    AI_1 = (GS_1 * FI)

    a_1 = real(ifft2(A_1))
    ai_1 = real(ifft2(AI_1))

    A_2 = GS_2 * FO
    AI_2 = GS_2 * FI

    a_2 = real(ifft2(A_2))
    ai_2 = real(ifft2(AI_2))

    A_3 = GS_3 * FO
    AI_3 = GS_3 * FI

    a_3 = real(ifft2(A_3))
    ai_3 = real(ifft2(AI_3))

    A_4 = GS_4 * FO
    AI_4 = GS_4 * FI

    a_4 = real(ifft2(A_4))
    ai_4 = real(ifft2(AI_4))

    A_5 = GS_5 * FO
    AI_5 = GS_5 * FI

    a_5 = real(ifft2(A_5))
    ai_5 = real(ifft2(AI_5))
    del FO
    del FI

    del G_0
    del G_1
    del G_2
    del G_3
    del G_4
    del G_5

    del GS_0
    del GS_1
    del GS_2
    del GS_3
    del GS_4
    del GS_5
    c1 = ((a_1 / (l_0)))
    c2 = (a_2 / (l_0 + a_1))
    c3 = (a_3 / (l_0 + a_1 + a_2))
    c4 = (a_4 / (l_0 + a_1 + a_2 + a_3))
    c5 = (a_5 / (l_0 + a_1 + a_2 + a_3 + a_4))

    ci1 = (ai_1 / (li_0))
    ci2 = (ai_2 / (li_0 + ai_1))
    ci3 = (ai_3 / (li_0 + ai_1 + ai_2))
    ci4 = (ai_4 / (li_0 + ai_1 + ai_2 + ai_3))
    ci5 = (ai_5 / (li_0 + ai_1 + ai_2 + ai_3 + ai_4))

    d1 = ctf(2 / VA)
    d2 = ctf(4 / VA)
    d3 = ctf(8 / VA)
    d4 = ctf(16 / VA)
    d5 = ctf(32 / VA)

    ai_1 = cmaskn_modified(c1, ci1, a_1, ai_1, 1)
    ai_2 = cmaskn_modified(c2, ci2, a_2, ai_2, 2)
    ai_3 = cmaskn_modified(c3, ci3, a_3, ai_3, 3)
    ai_4 = cmaskn_modified(c4, ci4, a_4, ai_4, 4)
    ai_5 = cmaskn_modified(c5, ci5, a_5, ai_5, 5)

    A_1 = gthresh_modified(c1, d1, a_1)
    AI_1 = gthresh_modified(ci1, d1, ai_1)
    A_2 = gthresh_modified(c2, d2, a_2)
    AI_2 = gthresh_modified(ci2, d2, ai_2)
    A_3 = gthresh_modified(c3, d3, a_3)
    AI_3 = gthresh_modified(ci3, d3, ai_3)
    A_4 = gthresh_modified(c4, d4, a_4)
    AI_4 = gthresh_modified(ci4, d4, ai_4)
    A_5 = gthresh_modified(c5, d5, a_5)
    AI_5 = gthresh_modified(ci5, d5, ai_5)

    y1 = (A_1 + A_2 + A_3 + A_4 + A_5)
    y2 = (AI_1 + AI_2 + AI_3 + AI_4 + AI_5)

    square_err = (y1 - y2) * (y1 - y2)
    np_ = sum(sum(square_err))

    sp = sum(sum(y1 ** 2))

    return 10 * log10(sp / np_)
