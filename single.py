import matplotlib.pyplot as plt
import cv2
import numpy as np
import demodulation


# def main():
#     img_short = cv2.imread('single/IMG_7574.JPG', 0)
#     img_long = cv2.imread('single/IMG_7573.JPG', 0)
#     # plt.plot(np.sum(img_short, 0)/np.sum(img_long, 0))
#
#     h, v = img_short.shape
#     y, x = 3, 3
#     xstep = int(v / x)
#     ystep = int(h / y)
#     shorts = [[] for i in range(y)]
#     longs = [[] for i in range(y)]
#     layers = [[] for i in range(y)]
#     for i in range(y):
#         for j in range(x):
#             # cv2.namedWindow('X')
#             # cv2.imshow('X', img_short[ystep * j:ystep * (j + 1), xstep * i:xstep * (i + 1)])
#             # cv2.waitKeyEx(0)
#             shorts[i].append(np.sum(img_short[ystep * j:ystep * (j + 1), xstep * i:xstep * (i + 1)], 0))
#             longs[i].append(np.sum(img_long[ystep * j:ystep * (j + 1), xstep * i:xstep * (i + 1)], 0))
#             layers[i].append(shorts[i][j] / longs[i][j])
#     for i in range(y):
#         for j in range(x):
#             plt.subplot(y, x, i * y + j + 1)
#             plt.plot(layers[j][i])
#     plt.show()

def main1():
    img_short = cv2.imread('single/IMG_4798.JPG', 0)
    img_long = cv2.imread('single/IMG_4797.JPG', 0)
    # plt.plot(np.sum(img_short, 0)/np.sum(img_long, 0))

    h, v = img_short.shape
    y, x = 3, 3
    xstep = int(v / x)
    ystep = int(h / y)
    shorts = [[] for i in range(y)]
    longs = [[] for i in range(y)]
    layers = [[] for i in range(y)]
    for i in range(y):
        for j in range(x):
            # cv2.namedWindow('X')
            # cv2.imshow('X', img_short[ystep * j:ystep * (j + 1), xstep * i:xstep * (i + 1)])
            # cv2.waitKeyEx(0)
            shorts[i].append(np.sum(img_short[ystep * j:ystep * (j + 1), xstep * i:xstep * (i + 1)], 0))
            longs[i].append(np.sum(img_long[ystep * j:ystep * (j + 1), xstep * i:xstep * (i + 1)], 0))
            layers[i].append(shorts[i][j] / longs[i][j])
    for i in range(y):
        for j in range(x):
            plt.subplot(y, x, i * y + j + 1)
            plt.plot(layers[j][i])
    plt.show()

def main2():
    img_short = cv2.imread('single/IMG_4798.JPG', 0)
    img_long = cv2.imread('single/IMG_4797.JPG', 0)
    ratio = np.sum(img_short, 0)/np.sum(img_long, 0)
    x = np.array([i for i in range(len(ratio))])
    # plt.plot(np.sum(img_short, 0))
    # plt.plot(np.sum(img_long, 0))
    plt.plot(ratio)
    plt.show()

if __name__ == '__main__':
    main2()
