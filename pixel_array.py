import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy import linalg as LA


def cos(v1, v2):
    return np.dot(v1, v2) / (LA.norm(v1) * LA.norm(v2))


def stack(R, d):
    return np.vstack((np.hstack((R, -np.dot(R, d).reshape(3, 1))), np.array([0, 0, 0, 1])))


def area(p1, p2, p3, p4):
    return (p1[0] * p2[1] + p2[0] * p3[1] + p3[0] * p4[1] + p4[0] * p1[1] - p2[0] * p1[1] - p3[0] * p2[1] - p4[0] * p3[
        1] - p1[0] * p4[1]) / 2


def s2c(P, f, R, d):
    zc = (R[2, 2] * d[2]) / (R[2, 0] * P[0] / f + R[2, 1] * P[1] / f + R[2, 2])
    xc = zc * P[0] / f
    yc = zc * P[1] / f
    return np.array([xc, yc, zc])


def c2r(P, R, d):
    return np.dot(R, P - d)


def w2r(P, R1, d1, R2, d2):
    pc = np.dot(R2, P - d2)
    return c2r(pc, R1, d1)


def channel(m, n, x0, y0, dx, dy, f, R1, d1, R2, d2):
    H0 = np.zeros((m, n), np.float)
    H1 = np.zeros((m, n), np.float)
    pss = np.zeros((m + 1, n + 1, 2), np.float)
    prs = np.zeros((m + 1, n + 1, 3), np.float)
    A1 = dx * dy
    normr = np.array([0, 0, 1])
    norml = np.dot(R1, np.dot(R2, normr))
    normc = np.dot(R1, normr)
    ol = w2r(np.array([0, 0, 0]), R1, d1, R2, d2)
    oc = w2r(d2, R1, d1, R2, d2)
    for i in range(m + 1):
        for j in range(n + 1):
            pss[i, j] = np.multiply([i - x0, j - y0], [dx, dy])
            prs[i, j] = s2c(pss[i, j], f, R1, d1)
            prs[i, j] = c2r(prs[i, j], R1, d1)
    for i in range(m):
        for j in range(n):
            ps = np.multiply([i - x0 + 0.5, j - y0 + 0.5], [dx, dy])
            pc = s2c(ps, f, R1, d1)
            pr = c2r(pc, R1, d1)

            # print(pr)

            dd0 = pr - ol
            A0 = area(prs[i, j], prs[i + 1, j], prs[i + 1, j + 1], prs[i, j + 1])
            H0[i, j] = A0 / (np.pi * LA.norm(dd0) * LA.norm(dd0)) * cos(norml, dd0) * cos(normr, dd0)
            dd1 = oc - pr
            H1[i, j] = A1 / (np.pi * LA.norm(dd1) * LA.norm(dd1)) * cos(-normr, dd1) * cos(-normc, dd1)
    return H0, H1


def main():
    img_short = cv2.imread('model/IMG_3495.JPG', 0)
    img_long = cv2.imread('model/IMG_3494.JPG', 0)
    img_short = img_short[1966:2066, :]
    img_long = img_long[1966:2066, :]
    m, n = img_long.shape
    m = 10

    fx = 3283.30513826360
    fy = 3307.61952765632
    x0 = 1987.78303129894
    y0 = 1490.79756116532
    f = 3e-3
    dx = f / fx
    dy = f / fy

    # Oc1 --> Or1
    # R2 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # W 2 C
    # d2 = np.array([0, 0, 1])
    # R1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # C 2 R
    # d1 = np.array([0, 0, 0.5])

    # Oc2 --> Or1
    # R2 = np.array([[np.sqrt(2) / 2, np.sqrt(2) / 2, 0], [0, 0, 1], [np.sqrt(2) / 2, -np.sqrt(2) / 2, 0]])  # W 2 C
    # d2 = np.array([0, 0.5, 1])
    # R1 = np.array([[np.sqrt(2) / 2, 0, -np.sqrt(2) / 2], [0, 1, 0], [np.sqrt(2) / 2, 0, np.sqrt(2) / 2]])  # C 2 R
    # d1 = np.array([0, 0, np.sqrt(2) / 2])

    # Oc1 -->Or2
    # R2 = np.array([[-np.sqrt(2) / 2, np.sqrt(2) / 2, 0], [0, 0, 1], [np.sqrt(2) / 2, np.sqrt(2) / 2, 0]])  # W 2 C
    # d2 = np.array([0, 0, 1])
    # R1 = np.array([[np.sqrt(2) / 2, 0, np.sqrt(2) / 2], [0, 1, 0], [-np.sqrt(2) / 2, 0, np.sqrt(2) / 2]])  # C 2 R
    # d1 = np.array([0, 0, np.sqrt(2) / 2])

    # Oc3 -->Or1
    R2 = np.array([[-np.sqrt(2) / 2, np.sqrt(2) / 2, 0], [1 / np.sqrt(6), 1 / np.sqrt(6), np.sqrt(6) / 3],
                   [1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)]])  # W 2 C
    d2 = np.array([0, -0.5, 1.5])
    R1 = np.array([[np.sqrt(2) / 2, 1 / np.sqrt(6), 1 / np.sqrt(3)], [0, np.sqrt(6) / 3, -1 / np.sqrt(3)],
                   [-np.sqrt(2) / 2, 1 / np.sqrt(6), 1 / np.sqrt(3)]])  # C 2 R
    d1 = np.array([0, 0, np.sqrt(3) / 2])

    H0, H1 = channel(m, n, x0, y0, dx, dy, f, R1, d1, R2, d2)
    Gy = np.sum(np.multiply(H0, H1), 0)

    Gy = Gy * np.sqrt(Gy[::-1]) * np.sqrt(np.sqrt(Gy[::-1]))
    Gy = Gy * 10e38

    markers_on = np.arange(0, 3024, 300).tolist()
    plt.figure(figsize=(4.5, 2.5))

    short = np.sum(img_short, 0)
    long = np.sum(img_long, 0)
    plt.plot(short, label='$i_{short}(y)$', marker='v', markevery=markers_on, markersize=4)
    plt.plot(long, label='$i_{long}(y)$', marker='s', markevery=markers_on, markersize=4)
    plt.plot(Gy[::-1], label='$G_y$', marker='o', markevery=markers_on, markersize=4)
    plt.legend(loc='upper center', ncol=3)
    plt.yticks([])
    plt.ylim(-400, 12000)
    # plt.show()
    plt.savefig('oc3or1.pdf', format='pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()

    # Oc1 --> Or1
    # R2 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # W 2 C
    # d2 = np.array([0, 0, 1])
    # R1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # C 2 R
    # d1 = np.array([0, 0, 0.5])
    # R0 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # W 2 R
    # d0 = np.array([0.5, 0, 1])

    # Oc2 --> Or1
    # R2 = np.array([[np.sqrt(2) / 2, np.sqrt(2) / 2, 0], [0, 0, 1], [np.sqrt(2) / 2, -np.sqrt(2) / 2, 0]])  # W 2 C
    # d2 = np.array([0, 0.5, 1])
    # R1 = np.array([[np.sqrt(2) / 2, 0, -np.sqrt(2) / 2], [0, 1, 0], [np.sqrt(2) / 2, 0, np.sqrt(2) / 2]])  # C 2 R
    # d1 = np.array([0, 0, np.sqrt(2) / 2])
    # R0 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # W 2 R
    # d0 = np.array([0.5, 0, 1])

    # Oc1 -->Or2
    # R2 = np.array([[-np.sqrt(2) / 2, np.sqrt(2) / 2, 0], [0, 0, 1], [np.sqrt(2) / 2, np.sqrt(2) / 2, 0]])  # W 2 C
    # d2 = np.array([0, 0, 1])
    # R1 = np.array([[np.sqrt(2) / 2, 0, np.sqrt(2) / 2], [0, 1, 0], [-np.sqrt(2) / 2, 0, np.sqrt(2) / 2]])  # C 2 R
    # d1 = np.array([0, 0, np.sqrt(2) / 2])
    # R0 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # W 2 R
    # d0 = np.array([0.5, 0.5, 1])

    # Oc3 -->Or1
    # R2 = np.array([[-np.sqrt(2) / 2, np.sqrt(2) / 2, 0], [1 / np.sqrt(6), 1 / np.sqrt(6), np.sqrt(6) / 3],
    #                [1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)]])  # W 2 C
    # d2 = np.array([0, -0.5, 1.5])
    # R1 = np.array([[np.sqrt(2) / 2, 1 / np.sqrt(6), 1 / np.sqrt(3)], [0, np.sqrt(6) / 3, -1 / np.sqrt(3)],
    #                [-np.sqrt(2) / 2, 1 / np.sqrt(6), 1 / np.sqrt(3)]])  # C 2 R
    # d1 = np.array([0, 0, np.sqrt(3) / 2])
    # R0 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # W 2 R
    # d0 = np.array([0.5, 0, 1])

    # print(stack(R0, d0))
    # print("-----------")
    # print(np.dot(stack(R1, d1), stack(R2, d2)))
