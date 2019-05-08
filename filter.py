import numpy as np
import demodulation


def first_derivative(ratio, N=7):
    f = []
    tmp = 0
    length = np.size(ratio)
    M = int((N - 1) / 2)
    m = int((N - 3) / 2)
    c = np.zeros(M)
    for k in np.arange(M):
        c[k] = (demodulation.combine(2 * m, m - k) - demodulation.combine(2 * m, m - k - 2)) / (2 ** (2 * m + 1))
    for i, v in enumerate(ratio):
        for j in np.arange(1, M + 1):
            tmp += ratio[i + j] * c[j - 1] if 0 <= i + j < length else ratio[i - j] * c[j - 1]
            tmp -= ratio[i - j] * c[j - 1] if 0 <= i - j < length else ratio[i + j] * c[j - 1]
        f.append(tmp)
        tmp = 0
    return f


def second_derivative(ratio, N=7):
    f = []
    tmp = 0
    length = np.size(ratio)
    M = int((N - 1) / 2)
    s = np.zeros(N)
    s[M:] = 0
    s[M] = 1
    for k in np.arange(M)[::-1]:
        s[k] = ((2 * N - 10) * s[k + 1] - (N + 2 * k + 3) * s[k + 2]) / (N - 2 * k - 1)
    for i, v in enumerate(ratio):
        tmp += ratio[i] * s[0]
        for j in np.arange(1, M + 1):
            tmp += ratio[i + j] * s[j] if 0 <= i + j < length else ratio[i - j] * s[j]
            tmp += ratio[i - j] * s[j] if 0 <= i - j < length else ratio[i + j] * s[j]
        tmp /= 2 ** (N - 3)
        f.append(tmp)
        tmp = 0
    return f


def high_pass_filter(ratio):
    ratio_freq = np.fft.rfft(ratio)
    ratio_freq[:2] = 0
    return np.fft.irfft(ratio_freq)
