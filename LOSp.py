# -*- coding: utf-8 -*-
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

def main():
    plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

    img_short = cv2.imread('data/1206/IMG_5591.JPG', 0)
    img_long = cv2.imread('data/1206/IMG_5590.JPG', 0)
    ratio = np.sum(img_short, 0)/np.sum(img_long, 0)
    plt.plot(np.sum(img_short, 0))
    plt.xlabel('Column')
    plt.ylabel('Brightness')
    # plt.plot(np.sum(img_long, 0))
    # plt.plot(ratio)
    plt.show()
    plt.savefig("FinalPaperimg/short.eps", format="eps")

main()