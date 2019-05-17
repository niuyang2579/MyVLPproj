import cv2
import numpy as np

def find_range():
    img_wide = cv2.imread('single/wide/IMG_1240.JPG', 0)
    img_tele = cv2.imread('single/tele/IMG_1242.JPG', 0)

    f = 3e-3
    # 第一次校准
    # fx_w = 3307.07231085391
    # fy_w = 3312.62054802747
    # x0_w = 2028.70804080261
    # y0_w = 1526.42411783125

    # fx_t = 6126.50690986855
    # fy_t = 6137.44825501589
    # x0_t = 1958.20904399740
    # y0_t = 1542.54884487071

    # 第二次校准
    fx_w = 3284.94667429144
    fy_w = 3284.37883290681
    x0_w = 2048.40202197768
    y0_w = 1537.40153756268

    fx_t = 6136.29226380575
    fy_t = 6122.83607411305
    x0_t = 2022.69956543486
    y0_t = 1458.12273021641

    dx_w = f / fx_w
    dy_w = f / fy_w

    dx_t = f / fx_t
    dy_t = f / fy_t

    m, n = img_tele.shape
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
    cropImg = img_wide[a:c, a:b]
    cv2.namedWindow("cropImg", 0)
    cv2.resizeWindow("cropImg", 480, 640)
    cv2.imshow('cropImg', cropImg)
    cv2.waitKey(0)

if __name__ == '__main__':
    find_range()