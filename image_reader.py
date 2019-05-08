import numpy as np
import urllib.request
import json
import cv2
from PIL import Image


def open_image(path):
    return Image.open(path).convert('L')


def sum_column(img, box):
    region = np.array(img.crop(box))
    return np.mean(region, 1)


def image(tag, index=-1):
    url = 'http://wsn.nwpu.info:3000/image/'
    # url = 'http://127.0.0.1:3000/image/'
    if index == -1:
        url = url + tag
    else:
        url = url + tag + '/' + str(index)
    with urllib.request.urlopen(url) as url:
        result = json.loads(url.read().decode())
        if result['success']:
            return result['result']['data']
        else:
            return False


def sub_sample(img, N):
    return img[int(N / 2)::N, int(N / 2)::N]


def segmentation(img, K):
    # change img(2D) to 1D
    img1 = img.reshape((img.shape[0] * img.shape[1], 1))
    img1 = np.float32(img1)

    # define criteria = (type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # set flags: hou to choose the initial center
    # ---cv2.KMEANS_PP_CENTERS ; cv2.KMEANS_RANDOM_CENTERS
    flags = cv2.KMEANS_RANDOM_CENTERS
    # apply kmenas
    compactness, labels, centers = cv2.kmeans(img1, K, None, criteria, 10, flags)
    img2 = labels.reshape(img.shape)
    return img2


def seg_ratio(img_short, img_long, labels, K, N):
    x, y = labels.shape
    seg_short = np.zeros((K, y * N))
    seg_long = np.zeros((K, y * N))

    for i in range(x):
        for j in range(y):
            for n in range(N):
                seg_short[labels[i, j], j * N + n] += np.sum(img_short[i * N:i * N + N, j * N + n])
                seg_long[labels[i, j], j * N + n] += np.sum(img_long[i * N:i * N + N, j * N + n])
    mask = seg_long >= np.max(seg_long, 0)
    ratio = seg_short.T[mask.T] / seg_long.T[mask.T]

    # ratios = np.zeros((K, y), np.float64)
    # counts = np.zeros(y)
    # for i in range(K):
    #     for j in range(y):
    #         if seg_long[i, j] != 0:
    #             ratios[i, j] = seg_short[i, j] / seg_long[i, j]
    #             counts[j] += 1
    # # ratio = np.mean(ratios, 0)
    # ratio = np.sum(ratios, 0) / counts
    return ratio[::-1]


def main():
    return


if __name__ == '__main__':
    main()
