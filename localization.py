import image_reader as ir
import matplotlib.pyplot as plt
import cv2
import demodulation
import numpy as np
import glob
import filter


# def main():
#     img_short = cv2.imread('case3/IMG_7092.JPG', 0)
#     img_long = cv2.imread('case3/IMG_7091.JPG', 0)
#     K = 5
#     N = 9
#     sub = ir.sub_sample(img_long, N)
#     labels = ir.segmentation(sub, K)
#     ratio = ir.seg_ratio(img_short, img_long, labels, K, N)
#     sig_layer = demodulation.first_derivative(ratio, 71)
#     plt.plot(sig_layer)
#     plt.show()


def main1():
    N = 51
    T = 16
    I = 9
    tag = '0117L21'
    i = 0
    img = ir.image(tag, i)

    img = filter.high_pass_filter(img)
    plt.subplot(211)
    plt.plot(img)

    rss = demodulation.extract_packets(img, T, N, I)
    print(rss)
    plt.subplot(212)
    plt.plot(img)

    plt.show()


def main2():
    img = ir.open_image('s300/IMG_7342.JPG')
    width, height = img.size
    box = (0, 0, width, height)
    iy = ir.sum_column(img, box)
    plt.plot(iy)
    # tag = 's300'
    # i = 6
    # plt.plot(ir.image(tag, i))
    # sig = []
    # for j in range(8):
    #     sig.append(ir.image(tag, j))
    #     sig[-1] = np.abs(np.fft.rfft(sig[-1]))
    #     sig[-1][:10] = 0
    # r = np.max(sig, 1)
    # print(r)
    # plt.plot(r)
    plt.show()


def main3():
    path = 's3001/角度由小变大'
    files = glob.glob(path + '/IMG*.' + 'JPG')
    files.sort()
    sig = []
    for file in files:
        print(file)
        img = ir.open_image(file)
        sig.append(np.mean(img))
    plt.plot(sig)
    plt.show()


def main4():
    tag = 'demo'
    i = 7
    img = ir.image(tag, i)
    plt.figure(figsize=(6, 2))
    plt.plot(img, label='$g(y)$')
    plt.plot([704, 704], [0.2, 1.2], linestyle="--", color='red')
    plt.plot([764, 764], [0.2, 1.2], linestyle="--", color='red')
    plt.plot([704, 764], [0.2, 0.2], linestyle="--", color='red')
    plt.plot([704, 764], [1.2, 1.2], linestyle="--", color='red')

    plt.plot(784, img[784], 'o', color='red')
    plt.plot([774, 794], [1.16, 1.16], linestyle="--", color='red')
    plt.annotate('', xy=(784, img[784]), xytext=(784, 1.16),
                 arrowprops=dict(arrowstyle="<->", connectionstyle="arc3,rad=0"))
    plt.annotate(r'$RSS_1$', xy=(784, img[784] + (1.16 - img[784]) / 2), xytext=(904, 0.8),
                 arrowprops=dict(arrowstyle="<-", connectionstyle="arc3,rad=0.1"))

    plt.plot([1688, 1688], [0.2, 1.2], linestyle="--", color='red')
    plt.plot([1747, 1747], [0.2, 1.2], linestyle="--", color='red')
    plt.plot([1688, 1747], [0.2, 0.2], linestyle="--", color='red')
    plt.plot([1688, 1747], [1.2, 1.2], linestyle="--", color='red')

    plt.xlim((650, 1800))
    plt.ylim((0.15, 1.55))
    plt.legend(loc='upper right')
    plt.xlabel('Row')
    plt.ylabel('Brightness')
    plt.show()
    # plt.savefig('yang5.pdf', format='pdf', bbox_inches='tight')


def main5():
    N = 51
    T = 16
    I = 6
    tag = 'neg1'
    i = 1
    LEDs = [[0.748, 2.349], [1.948, 2.349], [3.148, 2.349], [0.748, 0.549], [1.948, 0.549], [3.148, 0.549]]

    neg1 = (2.400, 3.305, 1.876, 285.000 / 180 * np.pi)
    neg2 = (2.400, 3.305, 1.867, 233.000 / 180 * np.pi)
    neg3 = (2.400, 3.305, 1.867, 310.000 / 180 * np.pi)
    neg4 = (1.8, 3.305, 1.857, 227 / 180 * np.pi)
    neg5 = (1.800, 3.305, 1.857, 315.000 / 180 * np.pi)
    neg6 = (1.200, 3.305, 1.860, 317.000 / 180 * np.pi)
    neg7 = (0.600, 0.905, 1.847, 0.000 / 180 * np.pi)
    neg8 = (4.200, 0.905, 1.833, 155.000 / 180 * np.pi)

    # m = - np.log(2) / np.log(np.cos(170 * np.pi / 360))
    # print(m)

    img = ir.image(tag, i)
    img = filter.high_pass_filter(img)

    rss = demodulation.extract_packets(img, T, N, I)
    pos = demodulation.neg2pos(rss)

    # simu_rss = []
    # simu_G = 1
    # for led in LEDs:
    #     simu_rss.append(demodulation.cal_rss(neg4 + (simu_G,), led, m))

    m = demodulation.calibrate_m(demodulation.f2, (1, 1), LEDs, pos[0], neg1)
    print(m)
    m = m['x'][1]

    K0 = (5, 5, 3.5, 2 * np.pi, 1)
    loc = demodulation.solve(demodulation.f, K0, LEDs, pos[0], m)
    loc['x'][3] = loc['x'][3] / np.pi * 180
    print(loc)


if __name__ == '__main__':
    main1()
