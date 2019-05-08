import numpy as np
import filter
from functools import reduce
import operator
from math import cos, sin
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def slope(point1, point2):
    return (point2[1] - point1[1]) / (point2[0] - point1[0])


def combine(n, k):
    if n == k or k == 0:
        return 1
    elif k < 0:
        return 0
    else:
        return reduce(operator.mul, range(n - k + 1, n + 1)) / reduce(operator.mul, range(1, k + 1))


def linear_regression(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m * x + c


def detect_preamble(ratio, N):
    preambles = []
    extremes = []
    plt.plot(ratio)
    plt.show()
    derivatives = filter.first_derivative(ratio, N)

    for i, d in enumerate(derivatives[1:]):
        if (d < 0) != (derivatives[i] < 0):
            extreme = [i, ratio[i]] if abs(d) > abs(derivatives[i]) else [i + 1, ratio[i + 1]]
            extremes.append(extreme)
    extremes = np.array(extremes)
    positive_mean = np.mean(ratio[ratio > 0])
    negative_mean = np.mean(ratio[ratio < 0])
    amplitude = (positive_mean - negative_mean) * 1.5
    slopes = []
    for i, extreme in enumerate(extremes[1:]):
        trend = 1 if extreme[1] >= extremes[i][1] else -1
        if trend * (extreme[1] - extremes[i][1]) > amplitude:
            slopes.append(abs(slope(extremes[i], extreme)))
    slope_coef = np.max(slopes) / 2
    count = 0
    for i, extreme in enumerate(extremes[1:]):
        trend = 1 if extreme[1] >= extremes[i][1] else -1
        if 2 * (count % 2) - 1 == trend:
            if trend * (extreme[1] - extremes[i][1]) > amplitude:
                if trend * slope(extremes[i], extreme) > slope_coef:
                    count += 1
                    if count == 3:
                        preambles.append((extremes[i - 2][0], extremes[i + 1][0]))
                        count = 0
            else:
                count = 0
        else:
            count = 0
    # filter outlier
    preambles = np.array(preambles)
    n_p = preambles[:, 1] - preambles[:, 0]
    d = np.abs(n_p - np.median(n_p))
    mdev = np.median(d)
    s = d / mdev if mdev else np.zeros(len(n_p))
    return preambles[s < 3]


def extract_packets(ratio, T, N, I):
    preambles = detect_preamble(ratio, N)
    n_p = preambles[:, 1] - preambles[:, 0]
    ni = np.mean(n_p) / 3
    packet_len = T * ni
    index = np.zeros(len(preambles))
    for i, preamble in enumerate(preambles[1:]):
        d = preamble[0] - (preambles[i][1] + ni)
        if d > packet_len / 2:
            d %= packet_len
            if d / ni < 4 or (packet_len - d) / ni < 4:
                index[i] = index[i + 1] = 1
    preambles = preambles[index > 0]
    d = preambles[-1][0] - preambles[0][0]
    s = int(round(d / (packet_len + 4 * ni)))
    ni = d / (s * (T + 4))
    rss = []
    for i in range(len(preambles) - 1):
        rss.append(np.zeros(I + 1))
        start = preambles[i][1]
        for j in range(I + 1):
            end = start + ni
            x = np.arange(int(round(start)), int(round(end)))
            y = ratio[x[0]:x[-1] + 1]
            ratio[x[0]:x[-1] + 1] = linear_regression(x, y)
            rss[-1][j] = ratio[x[-1]]
            if j != 0:
                rss[-1][j - 1] = (rss[-1][j - 1] + ratio[x[0]]) / 2
            start = end
        x = np.arange(int(round(start)), int(round(preambles[i + 1][0])))
        y = ratio[x[0]:x[-1] + 1]
        ratio[x[0]:x[-1] + 1] = linear_regression(x, y)
        rss[-1][I] = np.mean(ratio[x[0]:x[-1] + 1])
        plt.plot(ratio)
        plt.show()
    return rss


def neg2pos(rss):
    pos = []
    for i in range(len(rss)):
        rss[i] = rss[i][-1] - rss[i]
        pos.append(rss[i][:-1])
    return pos


def f(K, *args):
    LEDs = args[0]
    RSSs = args[1]
    m = args[2]
    x = K[0]
    y = K[1]
    z = K[2]
    theta = K[3]
    G = K[4]
    summation = 0
    for led, rss in zip(LEDs, RSSs):
        summation += ((G * (z ** m) * ((led[0] - x) * cos(theta) + (led[1] - y) * sin(theta))) / (
            ((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** ((m + 3) / 2)) - rss) ** 2
    return summation


def gradf(K, *args):
    LEDs = args[0]
    RSSs = args[1]
    m = args[2]
    x = K[0]
    y = K[1]
    z = K[2]
    theta = K[3]
    G = K[4]
    gx = 0
    gy = 0
    gz = 0
    gtheta = 0
    gG = 0
    for led, rss in zip(LEDs, RSSs):
        gx += 2 * ((G * (z ** m) * ((led[0] - x) * cos(theta) + (led[1] - y) * sin(theta))) / (
            ((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** ((m + 3) / 2)) - rss) * (
                  -G * (z ** m) * cos(theta) * (((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** ((m + 3) / 2)) - (
                      G * (z ** m) * ((led[0] - x) * cos(theta) + (led[1] - y) * sin(theta))) * ((m + 3) / 2) * (
                      ((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** ((m + 3) / 2 - 1)) * 2 * (x - led[0])) / (
                  ((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** (m + 3))
        gy += 2 * ((G * (z ** m) * ((led[0] - x) * cos(theta) + (led[1] - y) * sin(theta))) / (
            ((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** ((m + 3) / 2)) - rss) * (
                  -G * (z ** m) * sin(theta) * (((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** ((m + 3) / 2)) - (
                      G * (z ** m) * ((led[0] - x) * cos(theta) + (led[1] - y) * sin(theta))) * ((m + 3) / 2) * (
                      ((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** ((m + 3) / 2 - 1)) * 2 * (y - led[1])) / (
                  ((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** (m + 3))
        gz += 2 * ((G * (z ** m) * ((led[0] - x) * cos(theta) + (led[1] - y) * sin(theta))) / (
            ((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** ((m + 3) / 2)) - rss) * (
                  (m * G * (z ** (m - 1)) * ((led[0] - x) * cos(theta) + (led[1] - y) * sin(theta))) * (
                      ((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** ((m + 3) / 2)) - (
                      G * (z ** m) * ((led[0] - x) * cos(theta) + (led[1] - y) * sin(theta))) * ((m + 3) / 2) * (
                      ((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** ((m + 3) / 2 - 1)) * 2 * z) / (
                  ((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** (m + 3))
        gtheta += 2 * ((G * (z ** m) * ((led[0] - x) * cos(theta) + (led[1] - y) * sin(theta))) / (
            ((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** ((m + 3) / 2)) - rss) * (
                      G * (z ** m) * ((x - led[0]) * sin(theta) + (led[1] - y) * cos(theta))) / (
                      ((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** ((m + 3) / 2))

        gG += 2 * ((G * (z ** m) * ((led[0] - x) * cos(theta) + (led[1] - y) * sin(theta))) / (
            ((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** ((m + 3) / 2)) - rss) * (
                  z ** m * ((led[0] - x) * cos(theta) + (led[1] - y) * sin(theta))) / (
                  ((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** ((m + 3) / 2))
    return np.asarray((gx, gy, gz, gtheta, gG))


def solve(f, K0, *args):
    bnds = ((0, 5), (0, 5), (0, 3.5), (0, 2 * np.pi), (0, None))
    return minimize(f, K0, args=args, method='SLSQP', jac=gradf, bounds=bnds)


def cal_rss(K, led, m):
    x = K[0]
    y = K[1]
    z = K[2]
    theta = K[3]
    G = K[4]
    return (G * (z ** m) * ((led[0] - x) * cos(theta) + (led[1] - y) * sin(theta))) / (
        ((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** ((m + 3) / 2))


def f2(K, *args):
    LEDs = args[0]
    RSSs = args[1]
    loc = args[2]
    x = loc[0]
    y = loc[1]
    z = loc[2]
    theta = loc[3]
    G = K[0]
    m = K[1]
    summation = 0
    for led, rss in zip(LEDs, RSSs):
        summation += ((G * (z ** m) * ((led[0] - x) * cos(theta) + (led[1] - y) * sin(theta))) / (
            ((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** ((m + 3) / 2)) - rss) ** 2
    return summation


def gradf2(K, *args):
    LEDs = args[0]
    RSSs = args[1]
    loc = args[2]
    x = loc[0]
    y = loc[1]
    z = loc[2]
    theta = loc[3]
    G = K[0]
    m = K[1]

    gG = 0
    gm = 0
    for led, rss in zip(LEDs, RSSs):
        gG += 2 * ((G * (z ** m) * ((led[0] - x) * cos(theta) + (led[1] - y) * sin(theta))) / (
            ((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** ((m + 3) / 2)) - rss) * (
                  z ** m * ((led[0] - x) * cos(theta) + (led[1] - y) * sin(theta))) / (
                  ((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** ((m + 3) / 2))
        gm += 2 * ((G * (z ** m) * ((led[0] - x) * cos(theta) + (led[1] - y) * sin(theta))) / (
            ((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** ((m + 3) / 2)) - rss) * (
                  (np.log(m) * G * (z ** m) * ((led[0] - x) * cos(theta) + (led[1] - y) * sin(theta))) - (
                      G * (z ** m) * ((led[0] - x) * cos(theta) + (led[1] - y) * sin(theta))) * 1 / 2 * np.log(
                      (m + 3) / 2)) / (((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** ((m + 3) / 2))
    return np.asarray((gG, gm))


def calibrate_m(f2, K0, *args):
    return minimize(f2, K0, args=args, method='SLSQP', jac=gradf2, bounds=((0, None), (0, 1)))


def main():
    return


if __name__ == '__main__':
    main()
