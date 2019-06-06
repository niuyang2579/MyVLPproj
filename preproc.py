import cv2
import numpy as np
import matplotlib.pyplot as plt
import image_reader as ir
import demodulation
import filter
from math import sin, cos, degrees
from ExpData import data

def find_range():
    img_wide = cv2.imread('single/wide/IMG_1240.JPG', 0)
    img_tele = cv2.imread('single/tele/IMG_1242.JPG', 0)

    f = 3e-3
    # 第一次校准
    fx_w = 3307.07231085391
    fy_w = 3312.62054802747
    x0_w = 2028.70804080261
    y0_w = 1526.42411783125

    fx_t = 6126.50690986855
    fy_t = 6137.44825501589
    x0_t = 1958.20904399740
    y0_t = 1542.54884487071

    # 第二次校准
    # fx_w = 3284.94667429144
    # fy_w = 3284.37883290681
    # x0_w = 2048.40202197768
    # y0_w = 1537.40153756268
    #
    # fx_t = 6136.29226380575
    # fy_t = 6122.83607411305
    # x0_t = 2022.69956543486
    # y0_t = 1458.12273021641

    dx_w = f / fx_w
    dy_w = f / fy_w

    dx_t = f / fx_t
    dy_t = f / fy_t

    m, n = img_tele.shape
    print(m, n)
    p_t = [np.array([0, 0]), np.array([0, n-1]), np.array([m-1, 0]), np.array([m-1, n-1])]
    p_w = []
    #Ca, Cb是相机坐标系参数 Xc/Zc, Yc/Zc
    for point in p_t:
        x_w = ((point[0]-x0_t) / fx_t) * fx_w + x0_w
        y_w = ((point[1]-y0_t) / fy_t) * fy_w + y0_w
        p_w.append([x_w, y_w])
    a = int(p_w[0][0])
    b = int(p_w[1][1])
    c = int(p_w[2][0])
    # cropImg = img_wide[a:c, a:b]
    # cv2.namedWindow("cropImg", 0)
    # cv2.resizeWindow("cropImg", 480, 640)
    # cv2.imshow('cropImg', cropImg)
    # cv2.waitKey(0)
    return a, b, c

def observe():
    img_wide_l = cv2.imread('./images/0430dual/IMG_1141.JPG', 0)
    img_wide_s = cv2.imread('./images/0430dual/IMG_1142.JPG', 0)
    img_tele_l = cv2.imread('./images/0430dual/IMG_1143.JPG', 0)
    img_tele_s = cv2.imread('./images/0430dual/IMG_1144.JPG', 0)
    # a, b, c = find_range()
    # img_wide_l = img_wide_l[a:c, a:b]
    # img_wide_s = img_wide_s[a:c, a:b]
    plt.subplot(2, 1, 1)
    # plt.plot(np.sum(img_wide_s, 0)/np.sum(img_wide_l, 0))
    plt.plot(np.sum(img_wide_s, 0))
    plt.plot(np.sum(img_wide_l, 0))
    plt.subplot(2, 1, 2)
    # plt.plot(np.sum(img_tele_s, 0)/np.sum(img_tele_l, 0))
    plt.plot(np.sum(img_tele_s, 0))
    plt.plot(np.sum(img_tele_l, 0))
    plt.show()

def LCC():
    tag = '0529ED20'
    i = 0
    img = ir.image(tag, i)
    img = filter.high_pass_filter(img)
    # img_long = cv2.imread('./LCC/IMG_4847.JPG', 0)
    # img_short = cv2.imread('./LCC/IMG_4848.JPG', 0)
    plt.plot(img)
    plt.show()

def VLP(tag, K0, i):
    # tag = '0513P20'
    # K0 = [-1.193, 3.3465, 1, 10]
    # # K0 = [-2, 5, 2, 10]

    img = ir.image(tag, i)
    alpha = img[-1]  # yaw
    gamma = img[-2]  # pitch
    beta = img[-3]   # roll
    img = img[:(len(img)-3)]
    # 旋转矩阵
    R = np.array([[cos(alpha)*cos(beta), sin(alpha)*cos(beta), -sin(beta)],
                  [cos(alpha)*sin(beta)*sin(gamma)-sin(alpha)*cos(gamma), sin(alpha)*sin(beta)*sin(gamma)+cos(alpha)*cos(gamma), cos(beta)*sin(gamma)],
                  [cos(alpha)*sin(beta)*cos(gamma)+sin(alpha)*sin(gamma), sin(alpha)*sin(beta)*cos(gamma)-cos(alpha)*sin(gamma), cos(beta)*cos(gamma)]])
    I = np.array([3e-3, 3306.61902699479, 3295.67852676445, 2018.79741995009, 1469.65792747970])  # fs fx fy x0 y0
    indexes, RSSs = demodulation.cal_rss(img)
    LED = [-1.939, 3.638, 3.147]
    shape = [4032, 3024]
    P = demodulation.solve(demodulation.f4, demodulation.gradf4, K0, LED, RSSs, indexes, I, R, shape)['x'].tolist()
    print(tag, i, P)

def getDegree():
    tag = 'angle'
    i = 6
    img = ir.image(tag, i)
    alpha = img[-1]
    gamma = img[-2]
    beta = img[-3]
    print(degrees(alpha), degrees(beta), degrees(gamma))
    i = 7
    img = ir.image(tag, i)
    alpha = img[-1]
    gamma = img[-2]
    beta = img[-3]
    print(degrees(alpha), degrees(beta), degrees(gamma))

def prev(tag, i):
    img = ir.image(tag, i)
    img = img[:(len(img)-3)]
    # img = cv2.imread('./single/0602W.jpg', 0)
    img = filter.high_pass_filter(img)
    indexes, RSSs = demodulation.cal_rss(img)
    print(tag, i, len(indexes))
    # plt.plot(img)
    # plt.show()

def main():
    # for i in range(1, 15):
    #     tag = '052936V' + str(i)
    #     # print(tag)
    #     prev(tag, 1)
    # for i in range(7, 15):
    #     tag = '052935V' + str(i)
    #     VLP(tag, data[tag]['truth'], 1)
    tag = '052936V10'
    VLP(tag, data[tag]['truth'], 9)

if __name__ == '__main__':
    main()