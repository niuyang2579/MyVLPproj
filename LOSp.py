# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import image_reader as ir
import demodulation
import filter
import random
from math import sin, cos, degrees
from ExpData import data, trace_data, nlos_data, mnlos_data
from matplotlib.pyplot import MultipleLocator
import ast
import datetime

def main1():
    N = 51
    T = 16
    I = 6
    img_short = cv2.imread('data/0110/IMG_5697.JPG', 0)
    img_long = cv2.imread('data/0110/IMG_5696.JPG', 0)

    short = np.sum(img_short, 0)
    long = np.sum(img_long, 0)

    # ratio = np.fft.fft(short)
    ratio = np.sum(img_short, 0)/np.sum(img_long, 0)
    ratio = filter.high_pass_filter(ratio)
    rss = demodulation.extract_packets(ratio, T, N, I)

    # plt.plot(ratio)
    plt.xlabel('Column')
    plt.ylabel('Brightness')
    plt.savefig("FinalPaperimg/5-7.pdf", format="pdf", bbox_inches='tight')
    # plt.show()

    # plt.plot(ratio)
    # plt.show()
    # plt.plot(np.sum(img_long, 0))
    # plt.plot(ratio)

def main2():
    N = 51
    T = 16
    I = 9
    tag = '1018L1'
    i = 0
    img = ir.image(tag, i)
    # img = filter.high_pass_filter(img)
    # rss = demodulation.extract_packets(img, T, N, I)
    # print(rss)
    plt.plot(img)

    plt.savefig("FinalPaperimg/4-11.pdf", format="pdf", bbox_inches='tight')

    # plt.show()

def cal_error(target, truths):
    pass

def slos_trajectory():
    LED = [1.939, 3.638, 3.147]
    X = []
    Y = []
    Pret = np.loadtxt('./bpres/slos.txt')
    # Lret = np.loadtxt('./OptRes/Lret.txt')
    # ret = np.vstack((Pret, Lret))
    X_res = Pret[:, 0]
    Y_res = Pret[:, 1]
    X_res = -X_res
    print(Pret.shape)
    for i in range(1, 15):
        index = 'dot_' + str(i)
        X.append(trace_data[index]['truth'][0])
        Y.append(trace_data[index]['truth'][1])
        if len(X) == 1:
            continue
        # plt.annotate('', xy=(Y[-1], X[-1]), xytext=(Y[-2], X[-2]), arrowprops=dict(arrowstyle="->", color='C0'))
        plt.annotate('', xy=(X[-1], Y[-1]), xytext=(X[-2], Y[-2]))
    # print(X_res)
    fig = plt.figure(figsize=(6.89, 10.5))
    rect = [0.1, 0.1, 0.8, 0.8]
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    room = plt.imread('./FinalPaperimg/background.png')
    ax0.imshow(room)   #背景图片

    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    ax1.plot(X, Y, label='标记路径')
    plt.scatter(X_res, Y_res, c='C6', label='定位结果')
    plt.scatter(LED[0], LED[1], c='red', marker='s', label='LED位置')
    plt.xlim(0, 6.89)
    plt.ylim(0, 10.5)
    plt.legend(loc='best')
    plt.gca().invert_xaxis()
    plt.gca().yaxis.tick_right()
    # plt.savefig("FinalPaperimg/5-10.pdf", format="pdf", bbox_inches='tight')
    plt.show()

def slos_CDF():
    errors = []
    Pret = np.loadtxt('./bpres/slos.txt')
    truths = []
    for i in range(1, 15):
        index = 'dot_' + str(i)
        truths.append([trace_data[index]['truth'][0], trace_data[index]['truth'][1]])
    for res in Pret:
        errors.append(cal_error(res, truths))
    errors.sort()
    plotdata = [[], []]
    for i, error in enumerate(errors):
        plotdata[0].append(error)
        plotdata[1].append((i+1)/len(errors))
    plt.xlabel('error(m)', fontsize = 12)
    plt.ylabel('CDF', fontsize = 12)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.plot(plotdata[0], plotdata[1], '-', linewidth=2)
    plt.savefig("FinalPaperimg/5-11.pdf", format="pdf", bbox_inches='tight')
    for i in range(len(plotdata[0])):
        print(plotdata[0][i], plotdata[1][i])
    # plt.show()

def mlos_trajectory():
    LEDs = [[1.239, 3.638], [1.939, 3.638], [2.639, 3.638], [1.239, 2.938], [1.939, 2.938], [2.639, 2.938]]
    X = []
    Y = []
    Pret = np.loadtxt('./bpres/mlos.txt')
    # Lret = np.loadtxt('./OptRes/Lret.txt')
    # ret = np.vstack((Pret, Lret))
    X_res = Pret[:, 0]
    Y_res = Pret[:, 1]
    print(Pret.shape)
    for i in range(1, 37):
        index = 'dot_' + str(i)
        X.append(trace_data[index]['truth'][0])
        Y.append(trace_data[index]['truth'][1])
        if len(X) == 1:
            continue
        # plt.annotate('', xy=(Y[-1], X[-1]), xytext=(Y[-2], X[-2]), arrowprops=dict(arrowstyle="->", color='C0'))
        plt.annotate('', xy=(X[-1], Y[-1]), xytext=(X[-2], Y[-2]))
    # print(X_res)
    fig = plt.figure(figsize=(6.89, 10.5))
    rect = [0.1, 0.1, 0.8, 0.8]
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    room = plt.imread('./FinalPaperimg/background.png')
    ax0.imshow(room)   #背景图片

    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    ax1.plot(X, Y, label='标记路径')
    plt.scatter(X_res, Y_res, c='C6', label='定位结果')
    plt.scatter(LEDs[0][0], LEDs[0][1], c='red', marker='s', label='LED位置')
    for i in range(1, 6):
        plt.scatter(LEDs[i][0], LEDs[i][1], c='red', marker='s')
    plt.xlim(0, 6.89)
    plt.ylim(0, 10.5)
    plt.legend(loc='best')
    plt.gca().invert_xaxis()
    plt.gca().yaxis.tick_right()
    plt.savefig("FinalPaperimg/5-12.pdf", format="pdf", bbox_inches='tight')
    # plt.show()

def mlos_CDF():
    errors = []
    Pret = np.loadtxt('./bpres/mlos.txt')
    truths = []
    for i in range(1, 37):
        index = 'dot_' + str(i)
        truths.append([trace_data[index]['truth'][0], trace_data[index]['truth'][1]])
    for res in Pret:
        errors.append(cal_error(res, truths))
    errors.sort()
    plotdata = [[], []]
    for i, error in enumerate(errors):
        plotdata[0].append(error)
        plotdata[1].append((i+1)/len(errors))
    plt.xlabel('error(m)', fontsize = 12)
    plt.ylabel('CDF', fontsize = 12)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.plot(plotdata[0], plotdata[1], '-', linewidth=2)
    plt.savefig("FinalPaperimg/5-13.pdf", format="pdf", bbox_inches='tight')
    for i in range(len(plotdata[0])):
        print(plotdata[0][i], plotdata[1][i])
    # plt.show()

def snlos_trajectory():
    LED = [1.939, 3.638, 3.147]
    X = []
    Y = []
    Pret = np.loadtxt('./bpres/snlos.txt')
    X_res = Pret[:, 0]
    Y_res = Pret[:, 1]
    print(Pret.shape)
    for i in range(1, 8):
        index = 'dot_' + str(i)
        X.append(nlos_data[index]['truth'][0])
        Y.append(nlos_data[index]['truth'][1])
        if len(X) == 1:
            continue
        # plt.annotate('', xy=(Y[-1], X[-1]), xytext=(Y[-2], X[-2]), arrowprops=dict(arrowstyle="->", color='C0'))
        plt.annotate('', xy=(X[-1], Y[-1]), xytext=(X[-2], Y[-2]))
    fig = plt.figure(figsize=(6.89, 10.5))
    rect = [0.1, 0.1, 0.8, 0.8]
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    room = plt.imread('./FinalPaperimg/background.png')
    ax0.imshow(room)   #背景图片

    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    ax1.plot(X, Y, label='标记路径')
    plt.scatter(X_res, Y_res, c='C6', label='定位结果')
    plt.scatter(LED[0], LED[1], c='red', marker='s', label='LED位置')
    plt.xlim(0, 6.89)
    plt.ylim(0, 10.5)
    plt.legend(loc='best')
    plt.gca().invert_xaxis()
    plt.gca().yaxis.tick_right()
    # plt.savefig("FinalPaperimg/5-14.pdf", format="pdf", bbox_inches='tight')
    plt.show()

def snlos_CDF():
    errors = []
    Pret = np.loadtxt('./bpres/snlos.txt')
    truths = []
    for i in range(1, 8):
        index = 'dot_' + str(i)
        truths.append([nlos_data[index]['truth'][0], nlos_data[index]['truth'][1]])
    for res in Pret:
        errors.append(cal_error(res, truths))
    errors.sort()
    plotdata = [[], []]
    for i, error in enumerate(errors):
        plotdata[0].append(error)
        plotdata[1].append((i+1)/len(errors))
    plt.xlabel('error(m)', fontsize = 12)
    plt.ylabel('CDF', fontsize = 12)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.plot(plotdata[0], plotdata[1], '-', linewidth=2)
    # plt.savefig("FinalPaperimg/5-15.pdf", format="pdf", bbox_inches='tight')
    for i in range(len(plotdata[0])):
        print(plotdata[0][i], plotdata[1][i])
    plt.show()

def mnlos_trajectory():
    LEDs = [[1.239, 3.638], [1.939, 3.638], [2.639, 3.638], [1.239, 2.938], [1.939, 2.938], [2.639, 2.938]]
    X = []
    Y = []
    Pret = np.loadtxt('./bpres/mnlos.txt')
    X_res = Pret[:, 0]
    Y_res = Pret[:, 1]
    print(Pret.shape)
    for i in range(1, 17):
        index = 'dot_' + str(i)
        X.append(nlos_data[index]['truth'][0])
        Y.append(nlos_data[index]['truth'][1])
        if len(X) == 1:
            continue
        # plt.annotate('', xy=(Y[-1], X[-1]), xytext=(Y[-2], X[-2]), arrowprops=dict(arrowstyle="->", color='C0'))
        plt.annotate('', xy=(X[-1], Y[-1]), xytext=(X[-2], Y[-2]))
    fig = plt.figure(figsize=(6.89, 10.5))
    rect = [0.1, 0.1, 0.8, 0.8]
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    room = plt.imread('./FinalPaperimg/background.png')
    ax0.imshow(room)   #背景图片

    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    ax1.plot(X, Y, label='标记路径')
    plt.scatter(X_res, Y_res, c='C6', label='定位结果')
    plt.scatter(LEDs[0][0], LEDs[0][1], c='red', marker='s', label='LED位置')
    for i in range(1, 6):
        plt.scatter(LEDs[i][0], LEDs[i][1], c='red', marker='s')
    plt.xlim(0, 6.89)
    plt.ylim(0, 10.5)
    plt.legend(loc='best')
    plt.gca().invert_xaxis()
    plt.gca().yaxis.tick_right()
    # plt.savefig("FinalPaperimg/5-16.pdf", format="pdf", bbox_inches='tight')
    plt.show()

def mnlos_CDF():
    errors = []
    Pret = np.loadtxt('./bpres/mnlos.txt')
    truths = []
    for i in range(1, 17):
        index = 'dot_' + str(i)
        truths.append([mnlos_data[index]['truth'][0], mnlos_data[index]['truth'][1]])
    for res in Pret:
        errors.append(cal_error(res, truths))
    errors.sort()
    plotdata = [[], []]
    for i, error in enumerate(errors):
        plotdata[0].append(error)
        plotdata[1].append((i+1)/len(errors))
    plt.xlabel('error(m)', fontsize = 12)
    plt.ylabel('CDF', fontsize = 12)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.plot(plotdata[0], plotdata[1], '-', linewidth=2)
    plt.savefig("FinalPaperimg/5-17.pdf", format="pdf", bbox_inches='tight')
    for i in range(len(plotdata[0])):
        print(plotdata[0][i], plotdata[1][i])
    # plt.show()

# data_gernerate()
# mnlos_trajectory()
# mnlos_CDF()
# fix_data()
main2()