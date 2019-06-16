import cv2
import numpy as np
import matplotlib.pyplot as plt
import image_reader as ir
import demodulation
import filter
from math import sin, cos, degrees
from ExpData import data, trace_data
import ast
import datetime

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
    with open(r'result.log', 'a') as log_file:
        print(tag, i, P, file = log_file)

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

def trajectory():
    LED = [-1.939, 3.638, 3.147]
    plt.figure(figsize=(10.5 / 2, 6.89 / 2))
    plt.scatter(LED[0], LED[1], c='red', marker='s', label='LED locations')
    X = []
    Y = []
    Pret = np.loadtxt('./OptRes/Pret.txt')
    Lret = np.loadtxt('./OptRes/Lret.txt')
    ret = np.vstack((Pret, Lret))
    X_res = ret[:, 0]
    Y_res = ret[:, 1]
    print(ret.shape)
    for i in range(1, 29):
        index = 'dot_' + str(i)
        X.append(trace_data[index]['truth'][0])
        Y.append(trace_data[index]['truth'][1])
        if len(X) == 1:
            continue
        # plt.annotate('', xy=(Y[-1], X[-1]), xytext=(Y[-2], X[-2]), arrowprops=dict(arrowstyle="->", color='C0'))
        plt.annotate('', xy=(X[-1], Y[-1]), xytext=(X[-2], Y[-2]))
    # print(X_res)
    plt.plot(X, Y, label='Marked route')
    plt.scatter(X_res, Y_res, c='C1', label='Positioning results')
    plt.xlim(0, -6.89)
    plt.ylim(0, 10.5)
    plt.legend(loc='best')
    plt.gca().invert_xaxis()
    plt.gca().yaxis.tick_right()
    plt.show()

def main():
    # for i in range(1, 15):
    #     tag = '052936V' + str(i)
    #     # print(tag)
    #     prev(tag, 1)
    # for j in range(6, 11):
    #     for i in range(1, 21):
    #         tag = '0513P' + str(i)
    #         try:
    #             VLP(tag, data[tag]['truth'], j)
    #         except IndexError:
    #             with open(r'result.log', 'a') as log_file:
    #                 print(IndexError, tag, j, file = log_file)
    #         except:
    #             with open(r'result.log', 'a') as log_file:
    #                 print(tag, j, file = log_file)

    # for j in range(0, 11):
    #     for i in range(1, 21):
    #         tag = '0513L' + str(i)
    #         try:
    #             VLP(tag, data[tag]['truth'], j)
    #         except IndexError:
    #             with open(r'result.log', 'a') as log_file:
    #                 print(IndexError, tag, j, file = log_file)
    #         except:
    #             with open(r'result.log', 'a') as log_file:
    #                 print(tag, j, file = log_file)

    # for j in range(0, 11):
    #     for i in range(0, 9):
    #         tag = '0611D' + str(i)
    #         try:
    #             VLP(tag, data[tag]['non-truth'], j)
    #         except IndexError:
    #             with open(r'result.log', 'a') as log_file:
    #                 print(IndexError, tag, j, file = log_file)
    #         except:
    #             with open(r'result.log', 'a') as log_file:
    #                 print(tag, j, file = log_file)

    for j in range(0, 11):
        for i in range(0, 10):
            tag = '0616R' + str(5*i)
            try:
                VLP(tag, data[tag]['truth'], j)
            except IndexError:
                with open(r'result.log', 'a') as log_file:
                    print(IndexError, tag, i, file = log_file)
            except:
                with open(r'result.log', 'a') as log_file:
                    print(tag, i, file = log_file)

    # tag = '0513P7'
    # VLP(tag, data[tag]['truth'], 8)

def translation():
    with open('./OptRes/Lres.txt', 'r') as f:
        sourceInLine = f.readlines()
        Lres = []
        for line in sourceInLine:
            Lres.append(line.strip('\n').split('[')[1].split(']')[0].split(',')[0:2])
        for L in Lres:
            L[0] = float(L[0])
            L[1] = float(L[1]) + 3.6
        np.savetxt('./OptRes/Lret.txt', np.array(Lres))

def extract():
    with open('./OptRes/Dres.txt', 'r') as f:
        sourceInLine = f.readlines()
        res = []
        for line in sourceInLine:
            res.append(line.strip('\n').split('[')[1].split(']')[0].split(',')[0:2])
        for r in res:
            r[0] = float(r[0])
            r[1] = float(r[1])
        print(res)
        np.savetxt('./OptRes/Dret.txt', np.array(res))

def distance():
    errors3 = []
    errors6 = []
    errors9 = []
    n_bins = 100
    led = (-1.939, 3.638, 3.147)
    with open("OptRes/Dres2.txt") as f:
        for line in f.readlines():
            tag, index, coords = line.split(" ", 2)
            index = int(index)
            coords = ast.literal_eval(coords)
            truth = data[tag]['truth']
            error = np.sqrt((coords[0] - truth[0]) ** 2 + (coords[1] - truth[1]) ** 2)
            error *= 100
            if error > 100:
                continue
            d = np.sqrt((led[0] - truth[0]) ** 2 + (led[1] - truth[1]) ** 2)
            if 1 > d:
                errors3.append(error)
            elif 2 > d >= 1:
                errors6.append(error)
            elif d >= 2:
                errors9.append(error)

    errors3 = np.array(errors3)
    errors3 *= 2.5
    errors6 = np.array(errors6)
    errors6 *= 2.5
    errors9 = np.array(errors9)
    errors9 *= 2.5

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))

    counts, bin_edges = np.histogram(errors3, bins=n_bins)
    cdf = np.cumsum(counts)
    ax.plot(bin_edges[1:], cdf / cdf[-1], label='3m', linestyle='-')

    counts, bin_edges = np.histogram(errors6, bins=n_bins)
    cdf = np.cumsum(counts)
    ax.plot(bin_edges[1:], cdf / cdf[-1], label='6m', linestyle='--')

    counts, bin_edges = np.histogram(errors9, bins=n_bins)
    cdf = np.cumsum(counts)
    ax.plot(bin_edges[1:], cdf / cdf[-1], label='9m', linestyle=':')

    ax.grid(linestyle='--')
    ax.set_ylabel('CDF')
    ax.set_xlabel('Error in Euclidean distance (cm)')
    ax.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    # starttime = datetime.datetime.now()
    main()
    endtime = datetime.datetime.now()
    with open(r'result.log', 'a') as log_file:
        print(endtime)
    # trajectory()
    # translation()
    # extract()
    # distance()