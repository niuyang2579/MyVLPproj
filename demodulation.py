import numpy as np
import filter
from functools import reduce
import operator
from math import cos, sin
from scipy.optimize import minimize


def slope(point1, point2):
    return (point2[1] - point1[1]) / (point2[0] - point1[0])

# 组合数
def combine(n, k):
    if n == k or k == 0:
        return 1
    elif k < 0:
        return 0
    else:
        return reduce(operator.mul, range(n - k + 1, n + 1)) / reduce(operator.mul, range(1, k + 1))

# 线性规划
def linear_regression(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m * x + c


def detect_preamble(ratio, N):
    preambles = []
    extremes = []
    derivatives = filter.first_derivative(ratio, N)

    for i, d in enumerate(derivatives[1:]):
        if (d < 0) != (derivatives[i] < 0):
            extreme = [i, ratio[i]] if abs(d) > abs(derivatives[i]) else [i + 1, ratio[i + 1]]
            extremes.append(extreme)
    extremes = np.array(extremes)
    positive_mean = np.mean(ratio[ratio > 0])
    negative_mean = np.mean(ratio[ratio < 0])
    amplitude = (positive_mean - negative_mean) * 0.5
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
    return rss


def SNR(ratio, T, N, I):
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
    signal = 0
    noise = 0
    for i in range(len(preambles) - 1):
        start = preambles[i][1]
        for j in range(I + 1):
            end = start + ni
            x = np.arange(int(round(start)), int(round(end)))
            y = ratio[x[0]:x[-1] + 1]
            signal += np.sum(np.square(y))
            noise += np.sum(np.square(y - linear_regression(x, y)))
            start = end
        x = np.arange(int(round(start)), int(round(preambles[i + 1][0])))
        y = ratio[x[0]:x[-1] + 1]
        signal += np.sum(np.square(y))
        noise += np.sum(np.square(y - linear_regression(x, y)))
    return 10 * np.log10(signal / noise)


def neg2pos(neg):
    pos = []
    for i in range(len(neg)):
        neg[i] = neg[i][-1] - neg[i]
        pos.append(neg[i][:-1])
    return pos


def distance(truth, LEDs):
    return np.sqrt((truth[0] - LEDs[:, 0]) ** 2 + (truth[1] - LEDs[:, 1]) ** 2 + truth[2] ** 2)

# 入射角
def cos_phi(truth, LEDs):
    return truth[2] / distance(truth, LEDs)

# 发射角？
def cos_psi(truth, LEDs):
    return ((LEDs[:, 0] - truth[0]) * cos(truth[3]) + (LEDs[:, 1] - truth[1]) * sin(truth[3])) / distance(truth, LEDs)


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


def solve(f, gf, K0, *args):
    # bnds = ((0, 11), (0, 7), (0, 3.5), (0, 2 * np.pi), (0, None))
    bnds = ((0, 11), (0, 7), (0, 3.5), (0, None))
    return minimize(f, K0, args=args, method='SLSQP', jac=gf, bounds=bnds)


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


def f3(K, *args):
    LEDs = args[0]
    RSSs = args[1]
    loc = args[2]
    x = loc[0]
    y = loc[1]
    z = loc[2]
    theta = loc[3]
    G = K[0]
    m = 1
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


def gradf3(K, *args):
    LEDs = args[0]
    RSSs = args[1]
    loc = args[2]
    x = loc[0]
    y = loc[1]
    z = loc[2]
    theta = loc[3]
    G = K[0]
    m = 1

    gG = 0
    for led, rss in zip(LEDs, RSSs):
        gG += 2 * ((G * (z ** m) * ((led[0] - x) * cos(theta) + (led[1] - y) * sin(theta))) / (
            ((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** ((m + 3) / 2)) - rss) * (
                  z ** m * ((led[0] - x) * cos(theta) + (led[1] - y) * sin(theta))) / (
                  ((x - led[0]) ** 2 + (y - led[1]) ** 2 + z ** 2) ** ((m + 3) / 2))
    return np.asarray((gG,))


def calibrate_m(f2, K0, *args):
    return minimize(f2, K0, args=args, method='SLSQP', jac=gradf2, bounds=((0, None), (0, 1)))


def calibrate_G(f3, K0, *args):
    return minimize(f3, K0, args=args, method='SLSQP', jac=gradf3, bounds=((0, None),))

def f4(K, *args):
    xL, yL, zL = args[0]
    RSSs = args[1]
    indexes = args[2]
    fs, fx, fy, x0, y0 = args[3]
    R = args[4]
    m, n = args[5]
    dx = fs / fx
    dy = fs / fy

    xP = K[0]
    yP = K[1]
    zP = K[2]
    G = K[3]
    # xw = (((R[2, 0]*x + R[2, 1]*y + R[2, 2]*z)*(R[0, 0]*xs + R[0, 1]*ys + R[0, 2]*fs))/(
    #     R[2, 0]*xs + R[2, 1]*ys + R[2, 2]*fs) - (R[0, 0]*x + R[0, 1]*y + R[0, 2]*z))
    # yw = (((R[2, 0]*x + R[2, 1]*y + R[2, 2]*z)*(R[1, 0]*xs + R[1, 1]*ys + R[1, 2]*fs))/(
    #     R[2, 0]*xs + R[2, 1]*ys + R[2, 2]*fs) - (R[1, 0]*x + R[1, 1]*y + R[1, 2]*z))
    summation = 0
    for y, rss in zip(indexes, RSSs):
        u = 0
        for x in range(m):
            u += (G*zL**2*((R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) - (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2))) - (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) + (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))))/(zL**2 + (xL + R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2 + (yL + R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2) - rss
        summation += u ** 2
    return summation

def gradf4(K, *args):
    xL, yL, zL = args[0]
    RSSs = args[1]
    indexes = args[2]
    fs, fx, fy, x0, y0 = args[3]
    R = args[4]
    m, n = args[5]
    dx = fs / fx
    dy = fs / fy

    xP = K[0]
    yP = K[1]
    zP = K[2]
    G = K[3]
    gxP = 0
    gyP = 0
    gzP = 0
    gG = 0
    counter = 0
    for y, rss in zip(indexes, RSSs):
        print(counter)
        counter = counter + 1
        u = 0
        guxP = 0
        guyP = 0
        guzP = 0
        guG = 0
        for x in range(m):
            u += (G*zL**2*((R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) - (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2))) - (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) + (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))))/(zL**2 + (xL + R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2 + (yL + R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2)

            guxP += (G*zL**2*((R[1, 0] - (R[2, 0]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) - (R[1, 0] - (R[2, 0]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2))) - (R[1, 0] - (R[2, 0]*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) + (R[1, 0] - (R[2, 0]*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2))) + (R[0, 0] - (R[2, 0]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) - (R[0, 0] - (R[2, 0]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2))) - (R[0, 0] - (R[2, 0]*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) + (R[0, 0] - (R[2, 0]*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))))/(zL**2 + (xL + R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2 + (yL + R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2) - (G*zL**2*(2*(R[0, 0] - (R[2, 0]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(xL + R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) + 2*(R[1, 0] - (R[2, 0]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(yL + R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))))*((R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) - (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2))) - (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) + (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))))/(zL**2 + (xL + R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2 + (yL + R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2)**2

            guyP += (G*zL**2*((R[1, 1] - (R[2, 1]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) - (R[1, 1] - (R[2, 1]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2))) - (R[1, 1] - (R[2, 1]*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) + (R[1, 1] - (R[2, 1]*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2))) + (R[0, 1] - (R[2, 1]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) - (R[0, 1] - (R[2, 1]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2))) - (R[0, 1] - (R[2, 1]*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) + (R[0, 1] - (R[2, 1]*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))))/(zL**2 + (xL + R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2 + (yL + R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2) - (G*zL**2*(2*(R[0, 1] - (R[2, 1]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(xL + R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) + 2*(R[1, 1] - (R[2, 1]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(yL + R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))))*((R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) - (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2))) - (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) + (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))))/(zL**2 + (xL + R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2 + (yL + R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2)**2

            guzP += (G*zL**2*((R[1, 2] - (R[2, 2]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) - (R[1, 2] - (R[2, 2]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2))) - (R[1, 2] - (R[2, 2]*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) + (R[1, 2] - (R[2, 2]*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2))) + (R[0, 2] - (R[2, 2]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) - (R[0, 2] - (R[2, 2]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2))) - (R[0, 2] - (R[2, 2]*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) + (R[0, 2] - (R[2, 2]*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))))/(zL**2 + (xL + R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2 + (yL + R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2) - (G*zL**2*(2*(R[0, 2] - (R[2, 2]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(xL + R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) + 2*(R[1, 2] - (R[2, 2]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(yL + R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))))*((R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) - (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2))) - (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) + (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))))/(zL**2 + (xL + R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2 + (yL + R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2)**2

            guG += (zL**2*((R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) - (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1/2) + dy*R[0, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2))) - (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1/2) + dy*R[1, 1]*(y - y0 + 1/2)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1/2) + dy*R[2, 1]*(y - y0 + 1/2))) + (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[0, 0]*(x0 - x + 1/2) - fs*R[0, 2] + dy*R[0, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(dx*R[1, 0]*(x0 - x + 1/2) - fs*R[1, 2] + dy*R[1, 1]*(y0 - y + 1/2)))/(dx*R[2, 0]*(x0 - x + 1/2) - fs*R[2, 2] + dy*R[2, 1]*(y0 - y + 1/2)))))/(zL**2 + (xL + R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2 + (yL + R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2)

            # u += (G*zL**2*((R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) - (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1) + dy*R[1, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1))) - (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1) + dy*R[0, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) + (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1) + dy*R[0, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1) + dy*R[1, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))))/(zL**2 + (xL + R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2 + (yL + R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2)
            #
            # guxP += (G*zL**2*((R[1, 0] - (R[2, 0]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) - (R[1, 0] - (R[2, 0]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1) + dy*R[0, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1))) - (R[1, 0] - (R[2, 0]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1) + dy*R[1, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) + (R[1, 0] - (R[2, 0]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1) + dy*R[1, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1) + dy*R[0, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1))) + (R[0, 0] - (R[2, 0]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) - (R[0, 0] - (R[2, 0]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1) + dy*R[1, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1))) - (R[0, 0] - (R[2, 0]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1) + dy*R[0, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) + (R[0, 0] - (R[2, 0]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1) + dy*R[0, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1) + dy*R[1, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))))/(zL**2 + (xL + R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2 + (yL + R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2) - (G*zL**2*(2*(R[0, 0] - (R[2, 0]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(xL + R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) + 2*(R[1, 0] - (R[2, 0]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(yL + R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))))*((R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) - (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1) + dy*R[1, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1))) - (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1) + dy*R[0, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) + (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1) + dy*R[0, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1) + dy*R[1, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))))/(zL**2 + (xL + R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2 + (yL + R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2)**2
            #
            # guyP += (G*zL**2*((R[1, 1] - (R[2, 1]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) - (R[1, 1] - (R[2, 1]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1) + dy*R[0, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1))) - (R[1, 1] - (R[2, 1]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1) + dy*R[1, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) + (R[1, 1] - (R[2, 1]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1) + dy*R[1, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1) + dy*R[0, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1))) + (R[0, 1] - (R[2, 1]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) - (R[0, 1] - (R[2, 1]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1) + dy*R[1, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1))) - (R[0, 1] - (R[2, 1]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1) + dy*R[0, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) + (R[0, 1] - (R[2, 1]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1) + dy*R[0, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1) + dy*R[1, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))))/(zL**2 + (xL + R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2 + (yL + R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2) - (G*zL**2*(2*(R[0, 1] - (R[2, 1]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(xL + R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) + 2*(R[1, 1] - (R[2, 1]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(yL + R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))))*((R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) - (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1) + dy*R[1, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1))) - (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1) + dy*R[0, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) + (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1) + dy*R[0, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1) + dy*R[1, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))))/(zL**2 + (xL + R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2 + (yL + R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2)**2
            #
            # guzP += (G*zL**2*((R[1, 2] - (R[2, 2]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) - (R[1, 2] - (R[2, 2]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1) + dy*R[0, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1))) - (R[1, 2] - (R[2, 2]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1) + dy*R[1, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) + (R[1, 2] - (R[2, 2]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1) + dy*R[1, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))*(R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1) + dy*R[0, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1))) + (R[0, 2] - (R[2, 2]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) - (R[0, 2] - (R[2, 2]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1) + dy*R[1, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1))) - (R[0, 2] - (R[2, 2]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1) + dy*R[0, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) + (R[0, 2] - (R[2, 2]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1) + dy*R[0, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1) + dy*R[1, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))))/(zL**2 + (xL + R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2 + (yL + R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2) - (G*zL**2*(2*(R[0, 2] - (R[2, 2]*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(xL + R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) + 2*(R[1, 2] - (R[2, 2]*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(yL + R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))))*((R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) - (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1) + dy*R[1, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1))) - (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1) + dy*R[0, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) + (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1) + dy*R[0, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1) + dy*R[1, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))))/(zL**2 + (xL + R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2 + (yL + R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2)**2
            #
            # guG += (zL**2*((R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) - (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1) + dy*R[1, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1))) - (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1) + dy*R[0, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0))) + (R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0 + 1) + dy*R[0, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))*(R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0 + 1) + dy*R[1, 1]*(y - y0 + 1)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0 + 1) + dy*R[2, 1]*(y - y0 + 1)))))/(zL**2 + (xL + R[0, 0]*xP + R[0, 1]*yP + R[0, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[0, 2] + dx*R[0, 0]*(x - x0) + dy*R[0, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2 + (yL + R[1, 0]*xP + R[1, 1]*yP + R[1, 2]*zP - ((R[2, 0]*xP + R[2, 1]*yP + R[2, 2]*zP)*(fs*R[1, 2] + dx*R[1, 0]*(x - x0) + dy*R[1, 1]*(y - y0)))/(fs*R[2, 2] + dx*R[2, 0]*(x - x0) + dy*R[2, 1]*(y - y0)))**2) - rss

        gxP += 2 * (u - rss) * guxP
        gyP += 2 * (u - rss) * guxP
        gzP += 2 * (u - rss) * guzP
        gG += 2 * (u - rss) * guG
    return np.asarray((gxP, gyP, gzP, gG))

def find_inflexions(img):
    inflexions = []
    derivatives = filter.first_derivative(img, 101)
    for i, d in enumerate(derivatives[1:]):
        if (d < 0) != (derivatives[i] < 0):
            inflexion = [i, img[i]] if abs(d) > abs(derivatives[i]) else [i + 1, img[i + 1]]
            if i > 2 and len(img) - i > 3:
                inflexions.append(inflexion)
    return inflexions


def cal_rss(img):
    inflexions = find_inflexions(img)
    peaks = []
    start, end = 0, 0
    for p in inflexions:
        end = p[0]
        x = np.arange(int(round(start)), int(round(end)))
        y = img[x[0]:x[-1] + 1]
        img[x[0]:x[-1] + 1] = linear_regression(x, y)
        if start == 0:
            peaks.append(img[int(end - 1)])
        else:
            peaks[-1] = (img[int(start)] + peaks[-1]) / 2
            peaks.append(img[int(end - 1)])
        start = end
    end = len(img)
    x = np.arange(int(round(start)), int(round(end)))
    y = img[x[0]:x[-1] + 1]
    img[x[0]:x[-1] + 1] = linear_regression(x, y)
    peaks[-1] = (img[int(start)] + peaks[-1]) / 2
    if peaks[0] > peaks[1]:
        del peaks[0]
        del inflexions[0]
    if peaks[-1] > peaks[-2]:
        del peaks[-1]
        del inflexions[-1]
    peaks = np.array(peaks)
    inflexions = np.array(inflexions)
    indexes = inflexions[:, 0][1::2]
    rss = peaks[1::2] - (peaks[0:-1:2] + peaks[2::2]) / 2
    return indexes, rss

def main():
    return


if __name__ == '__main__':
    main()
