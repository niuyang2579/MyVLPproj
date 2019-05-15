import cv2
import numpy as np

def find_range():
    img_wide = cv2.imread('single/wide/IMG_1240.JPG', 0)
    img_tele = cv2.imread('single/tele/IMG_1242.JPG', 0)

    f = 3e-3
    fx_w = 3307.07231085391
    fy_w = 3312.62054802747
    x0_w = 2028.70804080261
    y0_w = 1526.42411783125
    dx_w = f / fx_w
    dy_w = f / fy_w

    fx_t = 6126.50690986855
    fy_t = 6137.44825501589
    x0_t = 1958.20904399740
    y0_t = 1542.54884487071
    dx_t = f / fx_t
    dy_t = f / fy_t

    m, n = img_tele.shape
    p_t = [np.array([0, 0]), np.array([0, n-1]), np.array([m-1, 0]), np.array([m-1, n-1])]
    p_w = []
    #Ca, Cb是相机坐标系参数 Xc/Zc, Yc/Zc
    for point in p_t:
        Ca = (point[0]-x0_t) / fx_t
        Cb = (point[1]-y0_t) / fy_t
        x_t = Ca*fx_w + x0_w
        y_t = Cb*fy_w + y0_w
        p_t.append([x_t, y_t])
    print(p_w)

if __name__ == '__main__':
    find_range()