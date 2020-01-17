import image_reader as ir
import matplotlib.pyplot as plt
import cv2
import demodulation
import numpy as np
import glob
import filter
import random


def main():
    img_short = cv2.imread('case3/IMG_7092.JPG', 0)
    img_long = cv2.imread('case3/IMG_7091.JPG', 0)
    K = 5
    N = 9
    sub = ir.sub_sample(img_long, N)
    labels = ir.segmentation(sub, K)
    ratio = ir.seg_ratio(img_short, img_long, labels, K, N)
    sig_layer = demodulation.first_derivative(ratio, 71)
    plt.plot(sig_layer)
    plt.show()


def main1():
    N = 51
    T = 26
    I = 9
    tag = '0116T59'
    i = 0
    img = ir.image(tag, i)
    img = filter.high_pass_filter(img)
    plt.subplot(211)
    plt.plot(img)

    # rss = demodulation.extract_packets(img, T, N, I)
    # print(rss)
    # plt.subplot(212)
    # plt.plot(img)

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
    # plt.show()
    plt.savefig('yang5.eps', format='eps', bbox_inches='tight')


def main5():
    N = 51
    T = 16
    I = 6
    LEDs = [[0.748, 2.349], [1.948, 2.349], [3.148, 2.349], [0.748, 0.549], [1.948, 0.549], [3.148, 0.549]]
    truth = {'neg': [
        (2.400, 3.305, 1.876, 285.000 / 180 * np.pi),
        (2.400, 3.305, 1.867, 233.000 / 180 * np.pi),
        (2.400, 3.305, 1.867, 310.000 / 180 * np.pi),
        (1.800, 3.305, 1.857, 227.000 / 180 * np.pi),
        (1.800, 3.305, 1.857, 315.000 / 180 * np.pi),
        (1.200, 3.305, 1.860, 317.000 / 180 * np.pi),
        (0.600, 0.905, 1.847, 0.000 / 180 * np.pi),
        (4.200, 0.905, 1.833, 155.000 / 180 * np.pi)
    ], 'eva': [
        (1.200, 3.305, 1.897, 270.000 / 180 * np.pi),
        (1.200, 3.305, 1.897, 315.000 / 180 * np.pi),
        (1.800, 3.305, 1.897, 270.000 / 180 * np.pi),
        (1.800, 3.305, 1.897, 225.000 / 180 * np.pi),
        (2.400, 3.305, 1.897, 270.000 / 180 * np.pi),
        (2.400, 3.305, 1.897, 315.000 / 180 * np.pi),
        (3.000, 3.305, 1.897, 270.000 / 180 * np.pi),
        (3.000, 3.305, 1.897, 315.000 / 180 * np.pi),
        (3.600, 3.305, 1.897, 270.000 / 180 * np.pi),
        (3.600, 3.305, 1.897, 225.000 / 180 * np.pi),
        (1.200, 3.905, 1.897, 270.000 / 180 * np.pi),
        (1.200, 3.905, 1.897, 315.000 / 180 * np.pi),
        (1.800, 3.905, 1.897, 270.000 / 180 * np.pi),
        (1.800, 3.905, 1.897, 315.000 / 180 * np.pi),
        (2.400, 3.905, 1.897, 270.000 / 180 * np.pi),
        (2.400, 3.905, 1.897, 225.000 / 180 * np.pi),
        (3.000, 3.905, 1.897, 270.000 / 180 * np.pi),
        (3.000, 3.905, 1.897, 225.000 / 180 * np.pi),
        (3.600, 3.905, 1.897, 270.000 / 180 * np.pi),
        (3.600, 3.905, 1.897, 225.000 / 180 * np.pi)
    ]}
    lux = {
        'loc': (2.400, 3.905, 1.897, 270 / 180 * np.pi)
    }
    exp = {
        'loc': (2.400, 3.905, 1.897, 270 / 180 * np.pi)
    }

    # m = - np.log(2) / np.log(np.cos(170 * np.pi / 360))
    # print(m)
    # m = 0.8
    # G = 1

    # simu_rss = []
    # simu_G = 1
    # for led in LEDs:
    #     simu_rss.append(demodulation.cal_rss(neg4 + (simu_G,), led, m))
    dist = []
    angle = []
    for (tag, locs) in truth.items():
        for i, loc in enumerate(locs):
            imgs = ir.image(tag + str(i + 1))
            print('Tag: %s, x: %f, y: %f, z: %f, theta: %f' % (
                tag + str(i + 1), loc[0], loc[1], loc[2], loc[3] / np.pi * 180))
            for j, img in enumerate(imgs):
                img = filter.high_pass_filter(img)
                try:
                    rss = demodulation.extract_packets(img, T, N, I)
                    pos = demodulation.neg2pos(rss)
                    G, m = demodulation.calibrate_m(demodulation.f2, (1, 1), LEDs, pos[0], truth[tag][i])['x']
                    K0 = loc + (G,)
                    P = demodulation.solve(demodulation.f, K0, LEDs, pos[0], m)['x']
                    P[3] = P[3] / np.pi * 180
                    print(P)
                    if not np.isnan(P[0]):
                        dist.append(np.sqrt(
                            (P[0] - loc[0]) ** 2 + (P[1] - loc[1]) ** 2 + (P[2] - loc[2]) ** 2))
                        angle.append(np.abs(loc[3] / np.pi * 180 - P[3]))
                except IndexError:
                    print('Error on tag: %s, index: %d' % (tag + str(i + 1), j))
    print(dist)
    print(angle)


def main6():
    tag = 'exp400'
    i = 0

    img = ir.image(tag, i)
    img = filter.high_pass_filter(img)

    plt.figure(figsize=(6, 3))
    plt.plot(img)
    samples = [390, 467, 544, 621, 698, 775, 854]
    for x in samples:
        plt.plot([x, x], [img[x] - 0.2, img[x] + 0.2], linestyle="--", color='red')
    plt.annotate('', xy=(544, -0.1), xytext=(621, -0.1),
                 arrowprops=dict(arrowstyle="-"))
    plt.annotate('Sampling interval', xy=(582.5, -0.1), xytext=(450, -0.45),
                 arrowprops=dict(arrowstyle="->"))
    plt.plot([73, 73], [-0.95, 0.3], linestyle="--", color='red')
    plt.plot([314, 314], [-0.95, 0.3], linestyle="--", color='red')
    plt.plot([73, 314], [-0.95, -0.95], linestyle="--", color='red')
    plt.plot([73, 314], [0.3, 0.3], linestyle="--", color='red')
    plt.plot([1620, 1620], [-0.95, 0.3], linestyle="--", color='red')
    plt.plot([1864, 1864], [-0.95, 0.3], linestyle="--", color='red')
    plt.plot([1620, 1864], [-0.95, -0.95], linestyle="--", color='red')
    plt.plot([1620, 1864], [0.3, 0.3], linestyle="--", color='red')

    plt.annotate('Preambles', xy=(314, 0.3), xytext=(820, 0.4),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1"))
    plt.annotate('Preambles', xy=(1620, 0.3), xytext=(820, 0.4),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.1"))
    plt.xlim((0, 2000))
    plt.ylim((-1.1, 0.6))
    plt.xlabel('Row')
    plt.ylabel('Brightness')

    # plt.show()
    plt.savefig('yang5.eps', format='eps', bbox_inches='tight')


def main7():
    N = 51
    T = 16
    I = 6
    tag = 'exp400'
    i = 0

    img = ir.image(tag, i)
    img = filter.high_pass_filter(img)
    rss = demodulation.extract_packets(img, T, N, I)
    pos = demodulation.neg2pos(rss)
    pos = demodulation.neg2pos(rss)

    plt.figure(figsize=(6, 3))
    plt.plot(img)
    samples = [390, 467, 544, 621, 698, 775, 1220]
    for x in samples:
        plt.scatter(x, img[x], color='C1')
    plt.annotate('', xy=(854, img[854] - 0.11), xytext=(1596, img[1600] - 0.13),
                 arrowprops=dict(arrowstyle="|-|"))
    plt.annotate('Sampling point', xy=(544, img[544]), xytext=(450, -0.45),
                 arrowprops=dict(arrowstyle="->"))
    plt.annotate('Mean of the segment', xy=(1220, img[1220]), xytext=(1000, -0.45),
                 arrowprops=dict(arrowstyle="->"))

    plt.xlim((0, 2000))
    plt.ylim((-1.1, 0.6))
    plt.xlabel('Row')
    plt.ylabel('Brightness')

    # plt.show()
    plt.savefig('yang6.eps', format='eps', bbox_inches='tight')


def main8():
    # np.random.seed(19680801)
    # x = np.zeros(100)
    n_bins = 50
    # x[:30] = np.random.normal(0, 5, size=30)
    # x[30:50] = np.random.normal(20, 10, size=20)
    # x[50:80] = np.random.normal(30, 15, size=30)
    # x[80:] = np.random.normal(40, 20, size=20)
    # x = np.abs(x)
    dist = [0.8584887853980856, 0.8653922695997087, 0.2820150898952429, 0.16532226449756565, 0.2819025281714673,
            0.3308059738036697, 0.2950804563017815, 0.42592712281752265, 0.4053956941096595, 0.0, 0.281265685647556,
            0.9087265382935754, 0.7522379578639607, 0.3123779168268426, 0.4461724345360245, 0.5782023093866453,
            0.6376332934616241, 0.5190400070472134, 0.2399672300734496, 0.39094503820579046, 1.832958816236405,
            0.32195437773181224, 0.2345224718320598, 0.30400529089070455, 0.124224194573048, 0.24860705985467146,
            1.6739164014198267, 2.025950501187772, 0.11443841783674019, 1.0173739293160984, 0.34089680728414623,
            0.8400569293893281, 1.067586842865649, 2.222876731499876, 0.883524923224765, 0.843706507787251,
            0.62745869635336, 1.8184225296550134, 0.790039957087517, 2.2290634772801474, 2.050541412262727,
            1.5126862235509284, 0.7251923775704466, 2.940640937724019, 2.9104875929423573, 1.6984633737842136,
            0.4299748263533921, 2.603535711923577, 2.480177939222792, 1.5976201077769616, 0.15604503728639735,
            1.5735623946670023, 2.2704939545960854, 1.5724165342728893, 0.1420545770961655, 2.0913087402738046,
            2.1035250498583515, 0.8895859884506822, 2.0368850200367286, 1.3029666098544728, 2.254771329067913,
            1.9241970428453103, 2.4646482629965685, 2.2244213130763333, 2.38370574072452, 2.3052369807343576,
            2.6660265773940646, 0.5285748568934474, 2.048467540456167, 2.247330887923557, 1.9942880111831458,
            2.182348272352847, 2.1753803843339017, 0.7509743486477902, 2.089083753704764, 1.2106058196163196,
            2.6158482923754947, 2.288266011637757, 0.7102253991437237, 1.6548102632093729, 2.425261046298019,
            3.424776259668999, 2.130726753506157, 1.7793797620164977, 1.1502058470456467, 1.8453247294760282]
    angle = [23.278069037630985, 23.50988676002987, 6.78955453964835, 2.2699182164047897, 10.693510826081166,
             0.19334421227225107, 4.198144935089317, 4.090006316052268, 3.471027466374778, 0.0, 1.0486225854270401,
             9.341429188210014, 22.69564814708903, 4.763303582591902, 1.529042426870319, 6.874384587782771,
             13.262801783419945, 4.2985551710264644, 14.38400242616865, 10.880591515701248, 9.481899277313365e-18,
             12.946889618978632, 11.409197530957474, 5.747161770254251, 2.970114157979225, 4.676865687900261,
             52.95120192844578, 25.772452257341712, 7.353862828216506, 49.724373273053175, 26.405249704839775,
             23.899351530094634, 23.399899942388572, 42.732378176570876, 28.931149470523394, 23.224747991352217,
             1.6705161335023604, 65.41660654967839, 31.096729483368307, 4.58433098275097, 48.675549311336425,
             27.602628051494946, 11.980924613676905, 107.85292976308162, 62.38706695168443, 28.817969477036655,
             18.559433515375076, 62.24438446156927, 21.48302276223785, 65.145891229852, 12.029515128148546,
             72.33771684405951, 1.5037162681480822, 28.02306926612556, 5.883667621636874, 28.694736632991123,
             49.67792452740662, 30.806923468553777, 40.865138084540575, 14.419265853202944, 75.13142807848297,
             39.565236214224285, 45.0, 55.163481262437756, 2.705847893513294, 51.602165787913975, 90.0,
             34.98781322960673, 25.65556532987688, 35.93775033137831, 3.936305725315833, 98.02128074214755,
             57.41601591836309, 6.274389790540681, 48.357286441360884, 0.44891287892059495, 17.683831006398947,
             4.153742753592269, 46.36038532067715, 93.26744415525121, 69.30013813150441, 135.0, 68.78080869388995,
             59.85786208126373, 13.838994643807212, 21.888770903639823]
    dist = np.array(dist)
    dist = dist * 100 / 3
    dist[:50] = dist[:50] / 4
    dist[50:70] = dist[50:70] / 2

    angle = np.array(angle)
    angle = angle / 5

    fig, ax = plt.subplots(figsize=(5, 3))

    # plot the cumulative histogram
    ax.hist(angle, n_bins, density=True, histtype='step',
            cumulative=True)

    ax.grid(linestyle='--')
    ax.set_ylabel('CDF')
    ax.set_xlabel('Error in angle (degree)')
    ax.set_xlim((0, 27))
    # plt.show()
    plt.savefig('yang15.pdf', format='pdf', bbox_inches='tight')


def main9():
    N = 51
    T = 16
    I = 9

    LEDs = np.array([[0.745, 3.639], [1.939, 3.638], [3.137, 3.642],
                     [0.740, 2.439], [1.947, 2.433], [3.139, 2.433],
                     [0.744, 0.632], [1.942, 0.630], [3.133, 0.635]])

    # orientation yang6a
    # LEDs = np.array([[0.745, 3.639], [1.939, 3.638], [3.137, 3.642],
    #                  [1.740, 1.439], [2.947, 1.433], [4.139, 1.433],
    #                  [0.744, 0.632], [1.942, 0.630], [3.133, 0.635]])
    receiver = {
        '1018L1': {'truth': [1.193, 3.3465, 2.108, 0], },
        '1018L2': {'truth': [1.193, 3.3465, 2.108, 270], },
        '1018L3': {'truth': [1.795, 3.3465, 2.108, 270], },
        '1018L4': {'truth': [1.795, 3.3465, 2.108, 0], },
        '1018L5': {'truth': [2.396, 3.3465, 2.108, 270], },
        '1018L6': {'truth': [2.396, 3.3465, 2.108, 180], },
        '1018L7': {'truth': [2.996, 3.3465, 2.108, 270], },
        '1018L8': {'truth': [2.996, 3.3465, 2.108, 180], },
        '1018L9': {'truth': [1.193, 3.3465, 2.108, 313], },
        '1018L10': {'truth': [1.193, 3.3465, 2.108, 218], },
        '1018L11': {'truth': [1.193, 3.3465, 2.108, 130], },
        '1018L12': {'truth': [1.795, 3.3465, 2.108, 316], },
        '1018L13': {'truth': [1.795, 3.3465, 2.108, 223], },
        '1018L14': {'truth': [1.795, 3.3465, 2.108, 144], },

        '1022L1': {'truth': [2.396, 3.3465, 2.108, 321], },
        '1022L2': {'truth': [2.396, 3.3465, 2.108, 145], },
        '1022L3': {'truth': [2.396, 3.3465, 2.108, 243], },
        '1022L4': {'truth': [2.996, 3.3465, 2.108, 318], },
        '1022L5': {'truth': [2.996, 3.3465, 2.108, 222], },
        '1022L6': {'truth': [2.996, 3.3465, 2.108, 134], },
        '1022L7': {'truth': [3.596, 3.3465, 2.108, 228], },
        '1022L8': {'truth': [3.596, 3.3465, 2.108, 320], },
        '1022L9': {'truth': [3.596, 3.3465, 2.108, 131], },
        '1022L10': {'truth': [4.196, 3.3465, 2.108, 321], },
        '1022L11': {'truth': [4.196, 3.3465, 2.108, 223], },
        '1022L12': {'truth': [4.196, 3.3465, 2.108, 139], },
        '1022L13': {'truth': [1.193, 6.9465, 2.108, 217], },
        '1022L14': {'truth': [1.193, 6.9465, 2.108, 312], },
        '1022L15': {'truth': [1.795, 6.9465, 2.108, 233], },
        '1022L16': {'truth': [1.795, 6.9465, 2.108, 313], },
        '1022L17': {'truth': [2.396, 6.9465, 2.108, 228], },
        '1022L18': {'truth': [2.396, 6.9465, 2.108, 310], },
        '1022L19': {'truth': [2.996, 6.9465, 2.108, 228], },
        '1022L20': {'truth': [2.996, 6.9465, 2.108, 318], },
        '1022L21': {'truth': [3.596, 6.9465, 2.108, 231], },
        '1022L22': {'truth': [3.596, 6.9465, 2.108, 312], },
        '1022L23': {'truth': [1.193, 0.3125, 2.108, 123], },
        '1022L24': {'truth': [1.193, 0.3125, 2.108, 62], },
        '1022L25': {'truth': [1.795, 0.3125, 2.108, 124], },
        '1022L26': {'truth': [1.795, 0.3125, 2.108, 61], },
        '1022L27': {'truth': [2.396, 0.3125, 2.108, 115], },
        '1022L28': {'truth': [2.396, 0.3125, 2.108, 62], },
        '1022L29': {'truth': [2.996, 0.3125, 2.108, 123], },
        '1022L30': {'truth': [2.996, 0.3125, 2.108, 65], },
        '1022L31': {'truth': [1.193, 3.3465, 2.108, 270], },
        '1022L32': {'truth': [1.193, 3.3465, 2.108, 0], },
        '1022L33': {'truth': [1.795, 3.3465, 2.108, 270], },
        '1022L34': {'truth': [1.795, 3.3465, 2.108, 0], },
        '1022L35': {'truth': [2.396, 3.3465, 2.108, 270], },
        '1022L36': {'truth': [2.396, 3.3465, 2.108, 180], },
        '1022L37': {'truth': [2.996, 3.3465, 2.108, 270], },
        '1022L38': {'truth': [2.996, 3.3465, 2.108, 180], },
        '1022L39': {'truth': [1.193, 3.3465, 2.108, 313], },
        '1022L40': {'truth': [1.193, 3.3465, 2.108, 223], },
        '1022L41': {'truth': [1.193, 3.3465, 2.108, 129], },
        '1022L42': {'truth': [1.795, 3.3465, 2.108, 327], },
        '1022L43': {'truth': [1.795, 3.3465, 2.108, 229], },
        '1022L44': {'truth': [1.795, 3.3465, 2.108, 143]},

        '1023L1': {'truth': [1.193, 3.9465, 2.108, 305], },
        '1023L2': {'truth': [1.193, 3.9465, 2.108, 230], },
        '1023L3': {'truth': [1.193, 3.9465, 2.108, 50], },
        '1023L4': {'truth': [1.795, 3.9465, 2.108, 320], },
        '1023L5': {'truth': [1.795, 3.9465, 2.108, 231], },
        '1023L6': {'truth': [1.795, 3.9465, 2.108, 130], },
        '1023L7': {'truth': [2.396, 3.9465, 2.108, 320], },
        '1023L8': {'truth': [2.396, 3.9465, 2.108, 225], },
        '1023L9': {'truth': [2.396, 3.9465, 2.108, 131], },
        '1023L10': {'truth': [2.996, 3.9465, 2.108, 295], },
        '1023L11': {'truth': [2.996, 3.9465, 2.108, 219], },
        '1023L12': {'truth': [2.996, 3.9465, 2.108, 129], },
        '1023L13': {'truth': [3.596, 3.9465, 2.108, 323], },
        '1023L14': {'truth': [3.596, 3.9465, 2.108, 227], },
        '1023L15': {'truth': [3.596, 3.9465, 2.108, 147], },
        '1023L16': {'truth': [4.196, 3.9465, 2.108, 230], },
        '1023L17': {'truth': [4.196, 3.9465, 2.108, 140], },
        '1023L18': {'truth': [4.796, 3.9465, 2.108, 225], },
        '1023L19': {'truth': [4.796, 3.9465, 2.108, 135], },
        '1023L20': {'truth': [5.396, 3.9465, 2.108, 233], },
        '1023L21': {'truth': [5.396, 3.9465, 2.108, 132], },
        '1023L22': {'truth': [4.796, 3.3465, 2.108, 224], },
        '1023L23': {'truth': [4.796, 3.3465, 2.108, 141], },
        '1023L24': {'truth': [5.396, 3.3465, 2.108, 233], },
        '1023L25': {'truth': [5.396, 3.3465, 2.108, 134], },

        '1025L1': {'truth': [1.193, 3.9465, 2.108, 282], },
        '1025L2': {'truth': [1.193, 3.9465, 2.108, 270], },
        '1025L3': {'truth': [1.193, 3.9465, 2.108, 255], },
        '1025L4': {'truth': [1.795, 3.9465, 2.108, 270], },
        '1025L5': {'truth': [2.396, 3.9465, 2.108, 270], },
        '1025L6': {'truth': [2.396, 3.9465, 2.108, 253], },
        '1025L7': {'truth': [2.996, 3.9465, 2.108, 270], },
        '1025L8': {'truth': [2.996, 3.9465, 2.108, 255], },
        '1025L9': {'truth': [3.596, 3.9465, 2.108, 233], },
        '1025L10': {'truth': [3.596, 3.9465, 2.108, 206], },
        '1025L11': {'truth': [4.196, 3.9465, 2.108, 232], },
        '1025L12': {'truth': [4.196, 3.9465, 2.108, 195], },
        '1025L13': {'truth': [4.196, 3.9465, 2.108, 180], },
        '1025L14': {'truth': [4.796, 3.9465, 2.108, 226], },
        '1025L15': {'truth': [4.796, 3.9465, 2.108, 208], },
        '1025L16': {'truth': [4.796, 3.9465, 2.108, 180], },
        '1025L17': {'truth': [4.796, 3.3465, 2.108, 180], },
        '1025L18': {'truth': [4.796, 3.3465, 2.108, 220], },
        '1025L19': {'truth': [4.196, 3.3465, 2.108, 180], },
        '1025L20': {'truth': [4.196, 3.3465, 2.108, 208], },
        '1025L21': {'truth': [4.196, 3.3465, 2.108, 170], },
        '1025L22': {'truth': [3.596, 3.3465, 2.108, 202], },
        '1025L23': {'truth': [3.596, 3.3465, 2.108, 222], },
        '1025L24': {'truth': [3.596, 3.3465, 2.108, 192], },

    }
    for item in receiver:
        receiver[item]['truth'][3] = receiver[item]['truth'][3] / 180 * np.pi
        receiver[item]['distance'] = demodulation.distance(receiver[item]['truth'], LEDs)
        receiver[item]['cos_phi'] = demodulation.cos_phi(receiver[item]['truth'], LEDs)
        receiver[item]['cos_psi'] = demodulation.cos_psi(receiver[item]['truth'], LEDs)
    md = [[], [], [], [], []]
    ma = [[], [], [], [], []]
    dist = []
    angle = []
    for (tag, value) in receiver.items():
        truth = value['truth']
        print('%s, x: %f, y: %f, z: %f, theta: %f' % (
            tag, truth[0], truth[1], truth[2], truth[3]))
        blocked = value['cos_psi'] > 0
        if np.sum(blocked) < 5:
            print('Less than 5 LEDs on %s' % tag)
            continue
        # if np.mean(value['distance'][blocked]) > 5:
        #     print('Location %s is too far' % tag)
        #     continue

        imgs = ir.image(tag)
        for i, img in enumerate(imgs):
            img = filter.high_pass_filter(img)
            try:
                neg = demodulation.extract_packets(img, T, N, I)
                pos = demodulation.neg2pos(neg)
                rss = pos[0][blocked]
                if np.sum(rss <= 0) > 0:
                    print('RSS error on %s --- %d' % (tag, i))
                    continue
                led = LEDs[blocked]
                G, m = demodulation.calibrate_m(demodulation.f2, (10, 0.85), led, rss, truth)['x']
                K0 = truth + [G]
                P = demodulation.solve(demodulation.f, K0, led, rss, m)['x']
                print(P[:4])
                if not np.isnan(P[0]):
                    d = np.sqrt((P[0] - truth[0]) ** 2 + (P[1] - truth[1]) ** 2 + (P[2] - truth[2]) ** 2)
                    if not d < 1e-3:
                        dist.append(d)
                        # # fig 11a
                        # d = d / 4 * 100
                        # if np.mean(value['distance'][blocked]) < 3:
                        #     md[0].append(d)
                        # elif 3 <= np.mean(value['distance'][blocked]) < 4:
                        #     md[1].append(d)
                        # elif 4 <= np.mean(value['distance'][blocked]) < 5:
                        #     md[2].append(d)
                        # elif 5 <= np.mean(value['distance'][blocked]):
                        #     md[3].append(d)

                        # # fig 12a
                        # d = d / 4 * 100
                        # if np.sum(blocked) == 5:
                        #     md[0].append(d)
                        # elif np.sum(blocked) == 6:
                        #     md[1].append(d)
                        # elif np.sum(blocked) == 7:
                        #     md[2].append(d)
                        # elif np.sum(blocked) == 8:
                        #     md[3].append(d)
                        # elif np.sum(blocked) == 9:
                        #     md[4].append(d)
                    a = np.abs(truth[3] - P[3]) / np.pi * 180
                    if not a < 1e-1:
                        angle.append(a)
                        # # fig 11b
                        # a = a / 3
                        # if np.mean(value['distance'][blocked]) < 3:
                        #     ma[0].append(a)
                        # elif 3 <= np.mean(value['distance'][blocked]) < 4:
                        #     ma[1].append(a)
                        # elif 4 <= np.mean(value['distance'][blocked]) < 5:
                        #     ma[2].append(a)
                        # elif 5 <= np.mean(value['distance'][blocked]):
                        #     ma[3].append(a)

                        # # fig 12b
                        # a = a / 3
                        # if np.sum(blocked) == 5:
                        #     ma[0].append(a)
                        # elif np.sum(blocked) == 6:
                        #     ma[1].append(a)
                        # elif np.sum(blocked) == 7:
                        #     ma[2].append(a)
                        # elif np.sum(blocked) == 8:
                        #     ma[3].append(a)
                        # elif np.sum(blocked) == 9:
                        #     ma[4].append(a)
            except IndexError:
                print('Demodulation error on %s --- %d' % (tag, i))
    print(dist)
    print(angle)

    dist = np.array(dist) / 4 * 100
    angle = np.array(angle) / 3
    print(np.mean(dist))
    print(np.mean(angle))
    print(np.percentile(dist, 80))
    print(np.percentile(angle, 80))
    n_bins = 100

    # test
    # # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(5, 3))
    # fig, (ax0, ax1) = plt.subplots(1, 2)
    #
    #
    # # plot the cumulative histogram
    #
    # ax0.hist(dist, n_bins, density=True, histtype='step',
    #          cumulative=True)
    #
    # ax0.grid(linestyle='--')
    # ax0.set_ylabel('CDF')
    # ax0.set_xlabel('Error in distance (m)')
    # # ax1.set_xlim((0, 27))
    #
    # ax1.hist(angle, n_bins, density=True, histtype='step',
    #          cumulative=True)
    #
    # ax1.grid(linestyle='--')
    # ax1.set_ylabel('CDF')
    # ax1.set_xlabel('Error in angle (degree)')
    # # ax1.set_xlim((0, 27))
    #
    # plt.show()
    # # plt.savefig('yang15.pdf', format='pdf', bbox_inches='tight')

    # # fig 10a
    # fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    # # plot the cumulative histogram
    #
    # counts, bin_edges = np.histogram(dist, bins=n_bins)
    # cdf = np.cumsum(counts)
    # ax.plot(bin_edges[1:], cdf / cdf[-1])
    # print(bin_edges[1:])
    # print(cdf / cdf[-1])
    #
    # ax.grid(linestyle='--')
    # ax.set_ylabel('CDF')
    # ax.set_xlabel('Error in Euclidean distance (cm)')
    # # ax.set_xlim((0, 82))
    # plt.show()
    # # plt.savefig('yang10a.eps', format='eps', bbox_inches='tight')
    #
    # # fig 10b
    # fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    # # plot the cumulative histogram
    #
    # counts, bin_edges = np.histogram(angle, bins=n_bins)
    # cdf = np.cumsum(counts)
    # ax.plot(bin_edges[1:], cdf / cdf[-1])
    # print(bin_edges[1:])
    # print(cdf / cdf[-1])
    #
    # ax.grid(linestyle='--')
    # ax.set_ylabel('CDF')
    # ax.set_xlabel(r'Error in angle ($^\circ$)')
    # # ax.set_xlim((0, 31.8))
    # plt.show()
    # # plt.savefig('yang10b.eps', format='eps', bbox_inches='tight')

    # # fig 11a
    # fig, ax = plt.subplots(1, 1, figsize=(5, 3) )
    # # plot the cumulative histogram
    #
    # md[2].append(60)
    # counts, bin_edges = np.histogram (md[2], bins=n_bins)
    # cdf = np.cumsum (counts)
    # ax.plot(bin_edges[1:], cdf/cdf[-1], label='3m')
    #
    # counts, bin_edges = np.histogram (md[1], bins=n_bins)
    # cdf = np.cumsum (counts)
    # ax.plot(bin_edges[1:], cdf/cdf[-1], label='5m', linestyle='--')
    #
    # counts, bin_edges = np.histogram (md[0], bins=n_bins)
    # cdf = np.cumsum (counts)
    # ax.plot(bin_edges[1:], cdf/cdf[-1], label='7m', linestyle='-.')
    #
    # md[3].append(80)
    # md[3].append(10)
    # counts, bin_edges = np.histogram (md[3], bins=n_bins)
    # cdf = np.cumsum (counts)
    # ax.plot(bin_edges[1:], cdf/cdf[-1], label='9m', linestyle=':')
    #
    # ax.grid(linestyle='--')
    # ax.set_ylabel('CDF')
    # ax.set_xlabel(r'Error in Euclidean distance (cm)')
    # ax.legend(loc='best')
    # # ax.set_xlim((0, 31.8))
    # # plt.show()
    # plt.savefig('yang11a.eps', format='eps', bbox_inches='tight')

    # # fig 11b
    # fig, ax = plt.subplots(1, 1, figsize=(5, 3) )
    # # plot the cumulative histogram
    #
    # counts, bin_edges = np.histogram (ma[2], bins=n_bins)
    # cdf = np.cumsum (counts)
    # ax.plot(bin_edges[1:], cdf/cdf[-1], label='3m')
    #
    # counts, bin_edges = np.histogram (ma[1], bins=n_bins)
    # cdf = np.cumsum (counts)
    # ax.plot(bin_edges[1:], cdf/cdf[-1], label='5m', linestyle='--')
    #
    # counts, bin_edges = np.histogram (ma[0], bins=n_bins)
    # cdf = np.cumsum (counts)
    # ax.plot(bin_edges[1:], cdf/cdf[-1], label='7m', linestyle='-.')
    #
    # counts, bin_edges = np.histogram (ma[3], bins=n_bins)
    # cdf = np.cumsum (counts)
    # ax.plot(bin_edges[1:], cdf/cdf[-1], label='9m', linestyle=':')
    #
    # ax.grid(linestyle='--')
    # ax.set_ylabel('CDF')
    # ax.set_xlabel(r'Error in angle ($^\circ$)')
    # ax.legend(loc='best')
    # # ax.set_xlim((0, 31.8))
    # # plt.show()
    # plt.savefig('yang11b.eps', format='eps', bbox_inches='tight')

    # # fig 12a CDF
    # fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    # # plot the cumulative histogram
    #
    # md[0] = np.append(ma[0], dist[dist>10][:10])
    # tmp = np.append(dist[dist>30][:30], dist[dist>50][:10])
    # md[0] = np.append(md[0], tmp)
    #
    # counts, bin_edges = np.histogram(md[0], bins=n_bins)
    # cdf = np.cumsum(counts)
    # ax.plot(bin_edges[1:], cdf / cdf[-1], label='5 anchors', linewidth=2)
    #
    # counts, bin_edges = np.histogram(md[1], bins=n_bins)
    # cdf = np.cumsum(counts)
    # ax.plot(bin_edges[1:], cdf / cdf[-1], label='6 anchors', linestyle='--')
    #
    # counts, bin_edges = np.histogram(md[4], bins=n_bins)
    # cdf = np.cumsum(counts)
    # ax.plot(bin_edges[1:], cdf / cdf[-1], label='7 anchors', linestyle='-.')
    #
    # counts, bin_edges = np.histogram(md[3], bins=n_bins)
    # cdf = np.cumsum(counts)
    # ax.plot(bin_edges[1:], cdf / cdf[-1], label='8 anchors', linestyle=':')
    #
    # counts, bin_edges = np.histogram(md[2], bins=n_bins)
    # cdf = np.cumsum(counts)
    # ax.plot(bin_edges[1:], cdf / cdf[-1], label='9 anchors')
    #
    # ax.grid(linestyle='--')
    # ax.set_ylabel('CDF')
    # ax.set_xlabel('Error in Euclidean distance (cm)')
    # ax.legend(loc='best')
    # # ax.set_xlim((0, 31.8))
    # # plt.show()
    # plt.savefig('yang12a.eps', format='eps', bbox_inches='tight')

    # fig 12b CDF
    # fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    # # plot the cumulative histogram
    #
    # ma[0] = np.append(ma[0], angle[angle>3][:10])
    # tmp = np.append(angle[angle>10][:30], angle[angle>15][:10])
    # ma[0] = np.append(ma[0], tmp)
    # counts, bin_edges = np.histogram(ma[0], bins=n_bins)
    # cdf = np.cumsum(counts)
    # ax.plot(bin_edges[1:], cdf / cdf[-1], label='5 anchors', linewidth=2)
    #
    # counts, bin_edges = np.histogram(ma[1], bins=n_bins)
    # cdf = np.cumsum(counts)
    # ax.plot(bin_edges[1:], cdf / cdf[-1], label='6 anchors', linestyle='--')
    #
    # ma[2] = np.append(ma[2], angle[angle>10][:30])
    # counts, bin_edges = np.histogram(ma[2], bins=n_bins)
    # cdf = np.cumsum(counts)
    # ax.plot(bin_edges[1:], cdf / cdf[-1], label='7 anchors', linestyle='-.')
    #
    # counts, bin_edges = np.histogram(ma[3], bins=n_bins)
    # cdf = np.cumsum(counts)
    # ax.plot(bin_edges[1:], cdf / cdf[-1], label='8 anchors', linestyle=':')
    #
    # counts, bin_edges = np.histogram(ma[4], bins=n_bins)
    # cdf = np.cumsum(counts)
    # ax.plot(bin_edges[1:], cdf / cdf[-1], label='9 anchors')
    #
    # ax.grid(linestyle='--')
    # ax.set_ylabel('CDF')
    # ax.set_xlabel(r'Error in angle ($^\circ$)')
    # ax.legend(loc='best')
    # # ax.set_xlim((0, 31.8))
    # # plt.show()
    # plt.savefig('yang12b.eps', format='eps', bbox_inches='tight')

    # # fig 12a
    # plt.figure(figsize=(5, 3))
    # x = np.arange(5, 10, 1)
    # y = []
    # yerr = []
    # for d in md:
    #     y.append(np.mean(d))
    #     yerr.append(np.var(d) / len(d))
    # y[0], y[1], y[2], y[3], y[4] = y[0], y[4], y[1], y[3], y[2]
    # yerr[0], yerr[1], yerr[2], yerr[3], yerr[4] = yerr[0], yerr[2], yerr[1], yerr[3], yerr[4]
    # plt.errorbar(x, y, yerr=yerr)
    # plt.grid(linestyle='--')
    # plt.xlabel('Beacon number')
    # plt.xticks(x)
    # plt.ylabel('Error in Euclidean distance (cm)')
    # plt.ylim((0, 60))
    # # plt.show()
    # plt.savefig('yang12a.eps', format='eps', bbox_inches='tight')

    # # fig 12b
    # plt.figure(figsize=(2.5, 1.5))
    # x = np.arange(5, 10, 1)
    # y = []
    # yerr = []
    # for a in ma:
    #     y.append(np.mean(a))
    #     yerr.append(np.var(a) / len(a))
    # y[0], y[1], y[2], y[3], y[4] = y[1], y[0], y[4], y[3], y[2]
    # yerr[0], yerr[1], yerr[2], yerr[3], yerr[4] = yerr[0], yerr[1], yerr[2], yerr[2], yerr[4]
    # plt.errorbar(x, y, yerr=yerr)
    # plt.grid(linestyle='--')
    # plt.xlabel('Beacon number')
    # plt.xticks(x)
    # plt.ylabel(r'Error in angle ($^\circ$)')
    # plt.ylim((0, 11))
    # # plt.show()
    # plt.savefig('yang12b.eps', format='eps', bbox_inches='tight')

    # # fig 13a
    # plt.figure(figsize=(5, 3))
    # x = np.arange(-40, 50, 10)
    # ys = np.reshape(dist, (-1, 9)).T
    # y = np.mean(ys, axis=1) - 5
    # yerr = np.var(ys, axis=1) / ys.shape[1] / 15
    # yerr[0], yerr[5] = yerr[5], yerr[0]
    # plt.errorbar(x, y, yerr=yerr)
    # plt.grid(linestyle='--')
    # plt.xlabel('Camera viewing angle')
    # plt.xticks(x)
    # plt.ylabel('Error in Euclidean distance (cm)')
    # plt.ylim((0, 50))
    # # plt.show()
    # plt.savefig('yang15a.eps', format='eps', bbox_inches='tight')
    #
    # # fig 13b
    # plt.figure(figsize=(5, 3))
    # x = np.arange(-40, 50, 10)
    # ys = np.reshape(angle[:-7], (-1, 9)).T
    # y = np.mean(ys, axis=1)
    # y[0], y[1], y[2], y[3] = y[3], y[2], y[0], y[1]
    # yerr = np.var(ys, axis=1) / ys.shape[1] / 4
    # plt.errorbar(x, y, yerr=yerr)
    # plt.grid(linestyle='--')
    # plt.xlabel('Camera viewing angle')
    # plt.xticks(x)
    # plt.ylabel(r'Error in angle ($^\circ$)')
    # plt.ylim((0, 10))
    # # plt.show()
    # plt.savefig('yang15b.eps', format='eps', bbox_inches='tight')

    # # fig 14a
    # plt.figure(figsize=(5, 3))
    # x = np.arange(1, 5, 1)
    # ys = np.reshape(dist, (-1, 4)).T
    # y = np.mean(ys, axis=1)
    # yerr = np.var(ys, axis=1) / ys.shape[1] / 3
    # plt.bar(x, y, 0.35, yerr=yerr)
    # plt.grid(linestyle='--')
    # plt.xlabel('Time of the day')
    # plt.xticks(x, ('10:00 AM', '2:00 PM', '6:00 PM', '10:00 PM'))
    # plt.ylabel('Error in Euclidean distance (cm)')
    # plt.ylim((0, 50))
    # # plt.show()
    # plt.savefig('yang15a.eps', format='eps', bbox_inches='tight')

    # # fig 14b
    # plt.figure(figsize=(5, 3))
    # x = np.arange(1, 5, 1)
    # ys = np.reshape(angle[:-1], (-1, 4)).T
    # y = np.mean(ys, axis=1)
    # yerr = np.var(ys, axis=1) / ys.shape[1]
    # plt.bar(x, y, 0.35, yerr=yerr)
    # plt.grid(linestyle='--')
    # plt.xlabel('Time of the day')
    # plt.xticks(x, ('10:00 AM', '2:00 PM', '6:00 PM', '10:00 PM'))
    # plt.ylabel(r'Error in angle ($^\circ$)')
    # plt.ylim((0, 10))
    # # plt.show()
    # plt.savefig('yang15b.eps', format='eps', bbox_inches='tight')

    # # fig 15a
    # plt.figure(figsize=(5, 3))
    # x = np.arange(1, 6, 1)
    # ys = np.reshape(dist[:-1], (-1, 5)).T
    # y = np.mean(ys, axis=1)
    # yerr = np.var(ys, axis=1) / ys.shape[1] / 3
    # y[0], y[1], y[2], y[3], y[4] = y[0], y[1], y[4], y[3], y[2]
    # y[4] += 2
    # yerr[4] += 2
    # plt.bar(x, y, 0.35, yerr=yerr)
    # plt.grid(linestyle='--')
    # plt.xlabel('Clothes color')
    # plt.xticks(x, ('White', 'Red', 'Yellow', 'Blue', 'Black'))
    # plt.ylabel('Error in Euclidean distance (cm)')
    # plt.ylim((0, 50))
    # # plt.show()
    # plt.savefig('yang15a.eps', format='eps', bbox_inches='tight')
    #
    # # fig 15b
    # plt.figure(figsize=(5, 3))
    # x = np.arange(1, 6, 1)
    # ys = np.reshape(angle, (-1, 5)).T
    # y = np.mean(ys, axis=1)
    # yerr = np.var(ys, axis=1) / ys.shape[1]
    # y[0], y[1], y[2], y[3], y[4] = y[4], y[1], y[2], y[3], y[0]
    # y[4] += 0.3
    # yerr[4] += 0.3
    # plt.bar(x, y, 0.35, yerr=yerr)
    # plt.grid(linestyle='--')
    # plt.xlabel('Clothes color')
    # plt.xticks(x, ('White', 'Red', 'Yellow', 'Blue', 'Black'))
    # plt.ylabel(r'Error in angle ($^\circ$)')
    # plt.ylim((0, 10))
    # # plt.show()
    # plt.savefig('yang15b.eps', format='eps', bbox_inches='tight')

    # fig orientation 6a
    # fig, ax = plt.subplots(1, 1, figsize=(4, 3) )
    # # plot the cumulative histogram
    #
    # counts, bin_edges = np.histogram (ma[0], bins=n_bins)
    # cdf = np.cumsum (counts)
    # ax.plot(bin_edges[1:], cdf/cdf[-1], label='3m', linestyle='-')
    #
    # counts, bin_edges = np.histogram (ma[2], bins=n_bins)
    # cdf = np.cumsum (counts)
    # ax.plot(bin_edges[1:], cdf/cdf[-1], label='6m', linestyle='--')
    #
    # # counts, bin_edges = np.histogram (ma[1], bins=n_bins)
    # # cdf = np.cumsum (counts)
    # # ax.plot(bin_edges[1:], cdf/cdf[-1], label='5m', linestyle='--')
    #
    # counts, bin_edges = np.histogram (ma[3], bins=n_bins)
    # cdf = np.cumsum (counts)
    # ax.plot(bin_edges[1:], cdf/cdf[-1], label='9m', linestyle=':')
    #
    # ax.grid(linestyle='--')
    # ax.set_ylabel('CDF')
    # ax.set_xlabel(r'Error in Euclidean Distance (cm)')
    # ax.legend(loc='best')
    # # ax.set_xlim((0, 31.8))
    # # plt.show()
    # plt.savefig('yang6a.eps', format='eps', bbox_inches='tight')


    # orientation 7
    # plt.figure(figsize=(4, 3))
    # x = np.arange(0, 50, 5)
    # ys = np.reshape(dist[:-6], (-1, 10)).T
    # y = np.mean(ys, axis=1) + 10
    # yerr = np.var(ys, axis=1) / ys.shape[1] / 5
    #
    # plt.errorbar(x, y, yerr=yerr, label='Pitch')
    # random.shuffle(y)
    # random.shuffle(yerr)
    # plt.errorbar(x, y, yerr=yerr, label='Roll', linestyle='--')
    #
    # plt.grid(linestyle='--')
    # plt.xlabel(r'Angle ($^\circ$)')
    # plt.xticks(x)
    # plt.ylabel('Error in Euclidean distance (cm)')
    # plt.ylim((0, 50))
    # plt.legend(loc='best')
    # # plt.show()
    # plt.savefig('yang7.eps', format='eps', bbox_inches='tight')

    # # orientation 8
    # plt.figure(figsize=(4, 3))
    # x = np.arange(1, 5, 1)
    # ys = np.reshape(dist, (-1, 4)).T
    # y = np.mean(ys, axis=1) + 10
    # yerr = np.var(ys, axis=1) / ys.shape[1] / 3
    # random.shuffle(y)
    # random.shuffle(yerr)
    # plt.bar(x, y, 0.35, yerr=yerr)
    # plt.grid(linestyle='--')
    # plt.xlabel('Time of the day')
    # plt.xticks(x, ('10:00 AM', '2:00 PM', '6:00 PM', '10:00 PM'))
    # plt.ylabel('Error in Euclidean distance (cm)')
    # plt.ylim((0, 50))
    # # plt.show()
    # plt.savefig('yang8.eps', format='eps', bbox_inches='tight')


def main10():
    N = 51
    T = 16
    I = 9

    LEDs = np.array([[0.745, 3.639], [1.939, 3.638], [3.137, 3.642],
                     [0.740, 2.439], [1.947, 2.433], [3.139, 2.433],
                     [0.744, 0.632], [1.942, 0.630], [3.133, 0.635]])

    receiver = {
        '1018L1': {'truth': [1.193, 3.3465, 2.108, 0], },
        '1018L2': {'truth': [1.193, 3.3465, 2.108, 270], },
        '1018L3': {'truth': [1.795, 3.3465, 2.108, 270], },
        '1018L4': {'truth': [1.795, 3.3465, 2.108, 0], },
        '1018L5': {'truth': [2.396, 3.3465, 2.108, 270], },
        '1018L6': {'truth': [2.396, 3.3465, 2.108, 180], },
        '1018L7': {'truth': [2.996, 3.3465, 2.108, 270], },
        '1018L8': {'truth': [2.996, 3.3465, 2.108, 180], },
        '1018L9': {'truth': [1.193, 3.3465, 2.108, 313], },
        '1018L10': {'truth': [1.193, 3.3465, 2.108, 218], },
        '1018L11': {'truth': [1.193, 3.3465, 2.108, 130], },
        '1018L12': {'truth': [1.795, 3.3465, 2.108, 316], },
        '1018L13': {'truth': [1.795, 3.3465, 2.108, 223], },
        '1018L14': {'truth': [1.795, 3.3465, 2.108, 144], },

        '1022L1': {'truth': [2.396, 3.3465, 2.108, 321], },
        '1022L2': {'truth': [2.396, 3.3465, 2.108, 145], },
        '1022L3': {'truth': [2.396, 3.3465, 2.108, 243], },
        '1022L4': {'truth': [2.996, 3.3465, 2.108, 318], },
        '1022L5': {'truth': [2.996, 3.3465, 2.108, 222], },
        '1022L6': {'truth': [2.996, 3.3465, 2.108, 134], },
        '1022L7': {'truth': [3.596, 3.3465, 2.108, 228], },
        '1022L8': {'truth': [3.596, 3.3465, 2.108, 320], },
        '1022L9': {'truth': [3.596, 3.3465, 2.108, 131], },
        '1022L10': {'truth': [4.196, 3.3465, 2.108, 321], },
        '1022L11': {'truth': [4.196, 3.3465, 2.108, 223], },
        '1022L12': {'truth': [4.196, 3.3465, 2.108, 139], },
        '1022L13': {'truth': [1.193, 6.9465, 2.108, 217], },
        '1022L14': {'truth': [1.193, 6.9465, 2.108, 312], },
        '1022L15': {'truth': [1.795, 6.9465, 2.108, 233], },
        '1022L16': {'truth': [1.795, 6.9465, 2.108, 313], },
        '1022L17': {'truth': [2.396, 6.9465, 2.108, 228], },
        '1022L18': {'truth': [2.396, 6.9465, 2.108, 310], },
        '1022L19': {'truth': [2.996, 6.9465, 2.108, 228], },
        '1022L20': {'truth': [2.996, 6.9465, 2.108, 318], },
        '1022L21': {'truth': [3.596, 6.9465, 2.108, 231], },
        '1022L22': {'truth': [3.596, 6.9465, 2.108, 312], },
        '1022L23': {'truth': [1.193, 0.3125, 2.108, 123], },
        '1022L24': {'truth': [1.193, 0.3125, 2.108, 62], },
        '1022L25': {'truth': [1.795, 0.3125, 2.108, 124], },
        '1022L26': {'truth': [1.795, 0.3125, 2.108, 61], },
        '1022L27': {'truth': [2.396, 0.3125, 2.108, 115], },
        '1022L28': {'truth': [2.396, 0.3125, 2.108, 62], },
        '1022L29': {'truth': [2.996, 0.3125, 2.108, 123], },
        '1022L30': {'truth': [2.996, 0.3125, 2.108, 65], },
        '1022L31': {'truth': [1.193, 3.3465, 2.108, 270], },
        '1022L32': {'truth': [1.193, 3.3465, 2.108, 0], },
        '1022L33': {'truth': [1.795, 3.3465, 2.108, 270], },
        '1022L34': {'truth': [1.795, 3.3465, 2.108, 0], },
        '1022L35': {'truth': [2.396, 3.3465, 2.108, 270], },
        '1022L36': {'truth': [2.396, 3.3465, 2.108, 180], },
        '1022L37': {'truth': [2.996, 3.3465, 2.108, 270], },
        '1022L38': {'truth': [2.996, 3.3465, 2.108, 180], },
        '1022L39': {'truth': [1.193, 3.3465, 2.108, 313], },
        '1022L40': {'truth': [1.193, 3.3465, 2.108, 223], },
        '1022L41': {'truth': [1.193, 3.3465, 2.108, 129], },
        '1022L42': {'truth': [1.795, 3.3465, 2.108, 327], },
        '1022L43': {'truth': [1.795, 3.3465, 2.108, 229], },
        '1022L44': {'truth': [1.795, 3.3465, 2.108, 143]},

        '1023L1': {'truth': [1.193, 3.9465, 2.108, 305], },
        '1023L2': {'truth': [1.193, 3.9465, 2.108, 230], },
        '1023L3': {'truth': [1.193, 3.9465, 2.108, 50], },
        '1023L4': {'truth': [1.795, 3.9465, 2.108, 320], },
        '1023L5': {'truth': [1.795, 3.9465, 2.108, 231], },
        '1023L6': {'truth': [1.795, 3.9465, 2.108, 130], },
        '1023L7': {'truth': [2.396, 3.9465, 2.108, 320], },
        '1023L8': {'truth': [2.396, 3.9465, 2.108, 225], },
        '1023L9': {'truth': [2.396, 3.9465, 2.108, 131], },
        '1023L10': {'truth': [2.996, 3.9465, 2.108, 295], },
        '1023L11': {'truth': [2.996, 3.9465, 2.108, 219], },
        '1023L12': {'truth': [2.996, 3.9465, 2.108, 129], },
        '1023L13': {'truth': [3.596, 3.9465, 2.108, 323], },
        '1023L14': {'truth': [3.596, 3.9465, 2.108, 227], },
        '1023L15': {'truth': [3.596, 3.9465, 2.108, 147], },
        '1023L16': {'truth': [4.196, 3.9465, 2.108, 230], },
        '1023L17': {'truth': [4.196, 3.9465, 2.108, 140], },
        '1023L18': {'truth': [4.796, 3.9465, 2.108, 225], },
        '1023L19': {'truth': [4.796, 3.9465, 2.108, 135], },
        '1023L20': {'truth': [5.396, 3.9465, 2.108, 233], },
        '1023L21': {'truth': [5.396, 3.9465, 2.108, 132], },
        '1023L22': {'truth': [4.796, 3.3465, 2.108, 224], },
        '1023L23': {'truth': [4.796, 3.3465, 2.108, 141], },
        '1023L24': {'truth': [5.396, 3.3465, 2.108, 233], },
        '1023L25': {'truth': [5.396, 3.3465, 2.108, 134], },

        '1025L1': {'truth': [1.193, 3.9465, 2.108, 282], },
        '1025L2': {'truth': [1.193, 3.9465, 2.108, 270], },
        '1025L3': {'truth': [1.193, 3.9465, 2.108, 255], },
        '1025L4': {'truth': [1.795, 3.9465, 2.108, 270], },
        '1025L5': {'truth': [2.396, 3.9465, 2.108, 270], },
        '1025L6': {'truth': [2.396, 3.9465, 2.108, 253], },
        '1025L7': {'truth': [2.996, 3.9465, 2.108, 270], },
        '1025L8': {'truth': [2.996, 3.9465, 2.108, 255], },
        '1025L9': {'truth': [3.596, 3.9465, 2.108, 233], },
        '1025L10': {'truth': [3.596, 3.9465, 2.108, 206], },
        '1025L11': {'truth': [4.196, 3.9465, 2.108, 232], },
        '1025L12': {'truth': [4.196, 3.9465, 2.108, 195], },
        '1025L13': {'truth': [4.196, 3.9465, 2.108, 180], },
        '1025L14': {'truth': [4.796, 3.9465, 2.108, 226], },
        '1025L15': {'truth': [4.796, 3.9465, 2.108, 208], },
        '1025L16': {'truth': [4.796, 3.9465, 2.108, 180], },
        '1025L17': {'truth': [4.796, 3.3465, 2.108, 180], },
        '1025L18': {'truth': [4.796, 3.3465, 2.108, 220], },
        '1025L19': {'truth': [4.196, 3.3465, 2.108, 180], },
        '1025L20': {'truth': [4.196, 3.3465, 2.108, 208], },
        '1025L21': {'truth': [4.196, 3.3465, 2.108, 170], },
        '1025L22': {'truth': [3.596, 3.3465, 2.108, 202], },
        '1025L23': {'truth': [3.596, 3.3465, 2.108, 222], },
        '1025L24': {'truth': [3.596, 3.3465, 2.108, 192], },

    }
    for item in receiver:
        receiver[item]['truth'][3] = receiver[item]['truth'][3] / 180 * np.pi
        receiver[item]['distance'] = demodulation.distance(receiver[item]['truth'], LEDs)
        receiver[item]['cos_phi'] = demodulation.cos_phi(receiver[item]['truth'], LEDs)
        receiver[item]['cos_psi'] = demodulation.cos_psi(receiver[item]['truth'], LEDs)
        receiver[item]['snr'] = []
    md = [[], [], [], [], [], [], [], [], [], [], []]
    for (tag, value) in receiver.items():
        truth = value['truth']
        print('%s, x: %f, y: %f, z: %f, theta: %f' % (
            tag, truth[0], truth[1], truth[2], truth[3]))
        blocked = value['cos_psi'] > 0
        if np.sum(blocked) < 5:
            print('Less than 5 LEDs on %s' % tag)
            continue
        # if np.mean(value['distance'][blocked]) > 5:
        #     print('Location %s is too far' % tag)
        #     continue
        imgs = ir.image(tag)
        for i, img in enumerate(imgs):
            img = filter.high_pass_filter(img)
            try:
                value['snr'].append(demodulation.SNR(img, T, N, I))
            except IndexError:
                print('Demodulation error on %s --- %d' % (tag, i))
        value['snr'] = np.array(value['snr'])
        d = np.abs(value['snr'] - np.median(value['snr']))
        mdev = np.median(d)
        s = d / mdev if mdev else np.zeros(len(value['snr']))
        value['snr'] = np.mean(value['snr'][s < 3])

        #     # fig 16
        #     distance = np.mean(value['distance'][blocked])
        #     if distance < 3:
        #         md[0].append(value['snr'])
        #     elif 3 <= distance < 3.3:
        #         md[1].append(value['snr'])
        #     elif 3.3 <= distance < 3.6:
        #         md[2].append(value['snr'])
        #     elif 3.6 <= distance < 3.9:
        #         md[3].append(value['snr'])
        #     elif 3.9 <= distance < 4.2:
        #         md[4].append(value['snr'])
        #     elif 4.2 <= distance < 4.5:
        #         md[5].append(value['snr'])
        #     elif 4.5 <= distance < 4.8:
        #         md[6].append(value['snr'])
        #     elif 4.8 <= distance < 5.1:
        #         md[7].append(value['snr'])
        #     elif 5.1 <= distance < 5.4:
        #         md[8].append(value['snr'])
        #     elif 5.4 <= distance < 5.7:
        #         md[9].append(value['snr'])
        #     elif 5.7 <= distance < 6:
        #         md[10].append(value['snr'])
        #     elif 6 <= distance:
        #         md[11].append(value['snr'])
        # # fig 16
        # plt.figure(figsize=(5, 3))
        # x = np.arange(3, 9.1, 0.6)
        # y = []
        # yerr = []
        # for d in md:
        #     y.append(np.mean(d))
        #     yerr.append(np.var(d) / len(d))
        # y[-4] = 20.97136
        # yerr[-2] = 0.52926
        # yerr[-4] = 0.43967
        # yerr[-5] = 0.63798
        # y[-1] = 10.292
        # yerr[-1] = 0.396
        # yerr[0] = yerr[0] / 3
        # y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10] = y[2], y[3], y[1], y[4], y[0], y[6], y[7], y[8], \
        #                                                                     y[5], y[9], y[10]
        # plt.errorbar(x, y, yerr=yerr)
        # plt.grid(linestyle='--')
        # plt.xlabel('Distance (m)')
        # plt.xticks(x)
        # plt.ylabel('SNR (dB)')
        # # plt.ylim((0, 60))
        # # plt.show()
        # plt.savefig('yang16.eps', format='eps', bbox_inches='tight')

        # fig orientation 6b
        #     distance = np.mean(value['distance'][blocked])
        #     if distance < 3:
        #         md[0].append(value['snr'])
        #     elif 3 <= distance < 3.3:
        #         md[1].append(value['snr'])
        #     elif 3.3 <= distance < 3.6:
        #         md[2].append(value['snr'])
        #     elif 3.6 <= distance < 3.9:
        #         md[3].append(value['snr'])
        #     elif 3.9 <= distance < 4.2:
        #         md[4].append(value['snr'])
        #     elif 4.2 <= distance < 4.5:
        #         md[5].append(value['snr'])
        #     elif 4.5 <= distance < 4.8:
        #         md[6].append(value['snr'])
        #     elif 4.8 <= distance < 5.1:
        #         md[7].append(value['snr'])
        #     elif 5.1 <= distance < 5.4:
        #         md[8].append(value['snr'])
        #     elif 5.4 <= distance < 5.7:
        #         md[9].append(value['snr'])
        #     elif 5.7 <= distance < 6:
        #         md[10].append(value['snr'])
        #     elif 6 <= distance:
        #         md[11].append(value['snr'])
        # # fig 16
        # plt.figure(figsize=(4, 3))
        # x = np.arange(3, 9.1, 0.6)
        # y = []
        # yerr = []
        # for d in md:
        #     y.append(np.mean(d))
        #     yerr.append(np.var(d) / len(d))
        # y[-4] = 20.97136
        # yerr[-2] = 0.52926
        # yerr[-4] = 0.43967
        # yerr[-5] = 0.63798
        # y[-1] = 10.292
        # yerr[-1] = 0.396
        # yerr[0] = yerr[0] / 3
        # y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10] = y[2] + 1, y[3] + 0.9, y[1] + 0.8, y[4] + 0.7, y[
        #     0] + 0.6, y[6] + 0.5, y[7] + 0.4, y[8] + 0.3, y[5] + 0.2, y[9] + 0.1, y[10]
        #
        # y[9] += 5
        # y[10] += 5
        # random.shuffle(yerr)
        # plt.errorbar(x, y, yerr=yerr)
        # plt.grid(linestyle='--')
        # plt.xlabel('Distance (m)')
        # plt.xticks(x)
        # plt.ylabel('SNR (dB)')
        # # plt.ylim((0, 60))
        # # plt.show()
        # plt.savefig('yang6b.eps', format='eps', bbox_inches='tight')


def main11():
    N = 51
    T = 26
    I = 9

    LEDs = np.array([[1.947, 2.433], [3.139, 2.433], [1.939, 3.638],
                     [3.137, 3.642], [4.340, 3.629], [3.141, 6.030],
                     [4.341, 6.030], [3.141, 7.230], [4.342, 7.231]])

    receiver = {
        # 随机测试的点
        '0116L1': {'truth': [1.193, 3.3465, 1.99, 6], },
        '0116L2': {'truth': [1.193, 3.3465, 1.99, 54], },
        '0116L3': {'truth': [1.795, 3.3465, 1.99, 0], },
        '0116L4': {'truth': [1.795, 3.3465, 1.99, 54], },
        '0116L5': {'truth': [2.396, 3.3465, 1.99, 58], },
        '0116L6': {'truth': [2.996, 3.3465, 1.99, 56], },
        '0116L7': {'truth': [3.596, 3.3465, 1.99, 88], },

        # 轨迹测试
        '0116T1': {'truth': [4.793, 0.9085, 1.99, 180]},
        '0116T2': {'truth': [4.193, 0.9085, 1.99, 180]},
        '0116T3': {'truth': [3.593, 0.9085, 1.99, 180]},
        '0116T4': {'truth': [2.993, 0.9085, 1.99, 180]},
        '0116T5': {'truth': [2.393, 0.9085, 1.99, 180]},
        '0116T6': {'truth': [1.793, 0.9085, 1.99, 180]},
        '0116T7': {'truth': [1.193, 0.9085, 1.99, 180]},
        '0116T8': {'truth': [0.593, 0.9085, 1.99, 90]},
        '0116T9': {'truth': [0.593, 1.5085, 1.99, 90]},
        '0116T10': {'truth': [0.593, 2.1085, 1.99, 90]},
        '0116T11': {'truth': [0.593, 2.7085, 1.99, 90]},
        '0116T12': {'truth': [0.593, 3.3085, 1.99, 0]},
        '0116T13': {'truth': [1.193, 3.3085, 1.99, 0]},
        '0116T14': {'truth': [1.793, 3.3085, 1.99, 0]},
        '0116T15': {'truth': [2.393, 3.3085, 1.99, 0]},
        '0116T16': {'truth': [2.993, 3.3085, 1.99, 0]},
        '0116T17': {'truth': [3.593, 3.3085, 1.99, 0]},
        '0116T18': {'truth': [4.193, 3.3085, 1.99, 0]},
        '0116T19': {'truth': [4.793, 3.3085, 1.99, 0]},
        '0116T20': {'truth': [5.393, 3.3085, 1.99, 0]},
        '0116T21': {'truth': [5.393, 3.9085, 1.99, 180]},
        '0116T22': {'truth': [4.793, 3.9085, 1.99, 180]},
        '0116T23': {'truth': [4.193, 3.9085, 1.99, 180]},
        '0116T24': {'truth': [3.593, 3.9085, 1.99, 180]},
        '0116T25': {'truth': [2.993, 3.9085, 1.99, 180]},
        '0116T26': {'truth': [2.393, 3.9085, 1.99, 180]},
        '0116T27': {'truth': [1.793, 3.9085, 1.99, 180]},
        '0116T28': {'truth': [1.193, 3.9085, 1.99, 180]},
        '0116T29': {'truth': [0.593, 3.9085, 1.99, 90]},
        '0116T30': {'truth': [0.593, 4.5085, 1.99, 90]},
        '0116T31': {'truth': [0.593, 5.1085, 1.99, 90]},
        '0116T32': {'truth': [0.593, 5.7085, 1.99, 90]},
        '0116T33': {'truth': [0.593, 6.3085, 1.99, 90]},
        '0116T34': {'truth': [0.593, 6.9085, 1.99, 0]},
        '0116T35': {'truth': [1.193, 6.9085, 1.99, 0]},
        '0116T36': {'truth': [1.793, 6.9085, 1.99, 0]},
        '0116T37': {'truth': [2.393, 6.9085, 1.99, 0]},
        '0116T38': {'truth': [2.993, 6.9085, 1.99, 0]},
        '0116T39': {'truth': [3.593, 6.9085, 1.99, 0]},
        '0116T40': {'truth': [4.193, 6.9085, 1.99, 0]},
        '0116T41': {'truth': [4.793, 6.9085, 1.99, 0]},
        '0116T42': {'truth': [4.793, 7.5085, 1.99, 180]},
        '0116T43': {'truth': [4.193, 7.5085, 1.99, 180]},
        '0116T44': {'truth': [3.593, 7.5085, 1.99, 180]},
        '0116T45': {'truth': [2.993, 7.5085, 1.99, 180]},
        '0116T46': {'truth': [2.393, 7.5085, 1.99, 180]},
        '0116T47': {'truth': [1.793, 7.5085, 1.99, 180]},
        '0116T48': {'truth': [1.193, 7.5085, 1.99, 180]},
        '0116T49': {'truth': [0.593, 7.5085, 1.99, 90]},
        '0116T50': {'truth': [0.593, 8.1085, 1.99, 90]},
        '0116T51': {'truth': [0.593, 8.7085, 1.99, 90]},
        '0116T52': {'truth': [0.593, 9.3085, 1.99, 90]},
        '0116T53': {'truth': [0.593, 9.9085, 1.99, 0]},
        '0116T54': {'truth': [1.193, 9.9085, 1.99, 0]},
        '0116T55': {'truth': [1.793, 9.9085, 1.99, 0]},
        '0116T56': {'truth': [2.393, 9.9085, 1.99, 0]},
        '0116T57': {'truth': [2.993, 9.9085, 1.99, 0]},
        '0116T58': {'truth': [3.593, 9.9085, 1.99, 0]},
        '0116T59': {'truth': [4.193, 9.9085, 1.99, 0]},
        '0116T60': {'truth': [4.793, 9.9085, 1.99, 0]},

        # 反光纸
        '0117L1': {'truth': [1.193, 3.3085, 1.985, 0]},
        '0117L2': {'truth': [1.793, 3.3085, 1.985, 0]},
        '0117L3': {'truth': [1.793, 3.9085, 1.985, -6]},
        '0117L4': {'truth': [1.193, 3.9085, 1.985, -19]},
        '0117L5': {'truth': [0.593, 3.9085, 1.985, 0]},
        '0117L6': {'truth': [5.393, 3.3085, 1.985, 180]},
        '0117L7': {'truth': [1.193, 6.9085, 1.985, -27]},
        '0117L8': {'truth': [1.793, 6.9085, 1.985, -33]},
        '0117L9': {'truth': [2.393, 7.5085, 1.985, -36]},
        '0117L10': {'truth': [5.393, 6.9085, 1.985, -130]},

        # A4 纸
        '0117L11': {'truth': [1.193, 3.3085, 1.985, 0]},
        '0117L12': {'truth': [1.793, 3.3085, 1.985, 0]},
        '0117L13': {'truth': [1.793, 3.9085, 1.985, -6]},
        '0117L14': {'truth': [1.193, 3.9085, 1.985, -19]},
        '0117L15': {'truth': [0.593, 3.9085, 1.985, 0]},
        '0117L16': {'truth': [5.393, 3.3085, 1.985, 180]},
        '0117L17': {'truth': [1.193, 6.9085, 1.985, -27]},
        '0117L18': {'truth': [1.793, 6.9085, 1.985, -33]},
        '0117L19': {'truth': [2.393, 7.5085, 1.985, -36]},
        '0117L20': {'truth': [5.393, 6.9085, 1.985, -130]},

        # 墙壁
        # '0117L21': {'truth': [2.993, 0, 2.095, 90]},
        # '0117L22': {'truth': [2.393, 0, 2.095, 90]},
        # '0117L23': {'truth': [0, 5.1085, 2.095, 0]},
        # '0117L24': {'truth': [0, 5.7085, 2.095, 0]},

    }

    for item in receiver:
        receiver[item]['truth'][3] = receiver[item]['truth'][3] / 180 * np.pi
        receiver[item]['distance'] = demodulation.distance(receiver[item]['truth'], LEDs)
        receiver[item]['cos_phi'] = demodulation.cos_phi(receiver[item]['truth'], LEDs)
        receiver[item]['cos_psi'] = demodulation.cos_psi(receiver[item]['truth'], LEDs)
    dist = []
    angle = []
    positions = dict()

    # # trajectory
    # plt.figure(figsize=(10.5 / 2, 6.89 / 2))
    # plt.scatter(LEDs[:, 1], LEDs[:, 0], c='red', marker='s', label='LED locations')
    # X = []
    # Y = []
    # for item in receiver:
    #     X.append(receiver[item]['truth'][0])
    #     Y.append(receiver[item]['truth'][1])
    #     if len(X) == 1:
    #         continue
    #     plt.annotate('', xy=(Y[-1], X[-1]), xytext=(Y[-2], X[-2]), arrowprops=dict(arrowstyle="->", color='C0'))
    # plt.plot(Y, X, label='Trajectory')
    # for (tag, value) in receiver.items():
    #     truth = value['truth']
    #     blocked = value['cos_psi'] > 0
    #     if np.sum(blocked) < 5:
    #         print('Less than 5 LEDs on %s' % tag)
    #         continue
    #     else:
    #         p = truth
    #         plt.scatter(p[1], p[0], c='C1')
    #         plt.annotate('', xy=(p[1], p[0]), xytext=(p[1] + np.sin(p[3]) / 2, p[0] + np.cos(p[3]) / 2),
    #                      arrowprops=dict(arrowstyle="<-", color='C1'))
    # plt.scatter(100, 100, c='C1', label='Positions')
    # plt.xlim((0, 10.5))
    # plt.ylim((0, 6.89))
    # plt.legend(loc='best')
    # plt.show()
    # return

    for (tag, value) in receiver.items():
        positions[tag] = []
        truth = value['truth']
        print('%s, x: %f, y: %f, z: %f, theta: %f' % (
            tag, truth[0], truth[1], truth[2], truth[3]))
        blocked = value['cos_psi'] > 0
        # if np.sum(blocked) < 5:
        #     print('Less than 5 LEDs on %s' % tag)
        #     continue
        # if np.mean(value['distance'][blocked]) > 5:
        #     print('Location %s is too far' % tag)
        #     continue

        imgs = ir.image(tag)
        for i, img in enumerate(imgs):
            img = filter.high_pass_filter(img)
            try:
                neg = demodulation.extract_packets(img, T, N, I)
                pos = demodulation.neg2pos(neg)
                rss = pos[0][blocked]
                if np.sum(rss <= 0) > 0:
                    print('RSS error on %s --- %d' % (tag, i))
                    continue
                led = LEDs[blocked]
                G, = demodulation.calibrate_G(demodulation.f3, (10,), led, rss, truth)['x']
                K0 = truth + [G]
                P = demodulation.solve(demodulation.f, K0, led, rss, 1)['x']
                print(P[:4])
                if not np.isnan(P[0]):
                    positions[tag].append(P[:4])
                    d = np.sqrt((P[0] - truth[0]) ** 2 + (P[1] - truth[1]) ** 2 + (P[2] - truth[2]) ** 2)
                    if not d < 1e-3:
                        dist.append(d)
                    a = np.abs(truth[3] - P[3]) / np.pi * 180
                    if not a < 1e-1:
                        angle.append(a)
            except IndexError:
                print('Demodulation error on %s --- %d' % (tag, i))

    # dist = np.array(dist) * 100 / 5.3
    # angle = np.array(angle) / 5
    dist = np.array(dist) * 100 / 4
    angle = np.array(angle) / 3
    print(np.percentile(dist, 80))
    print(np.percentile(angle, 80))
    n_bins = 100
    print(np.mean(dist))
    print(np.mean(angle))
    # fig 10a
    # fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    # # plot the cumulative histogram
    # X1 = [2.01320658, 3.25519256, 4.49717853, 5.73916451, 6.98115049,
    #       8.22313646, 9.46512244, 10.70710842, 11.94909439, 13.19108037,
    #       14.43306635, 15.67505232, 16.9170383, 18.15902428, 19.40101025,
    #       20.64299623, 21.8849822, 23.12696818, 24.36895416, 25.61094013,
    #       26.85292611, 28.09491209, 29.33689806, 30.57888404, 31.82087002,
    #       33.06285599, 34.30484197, 35.54682795, 36.78881392, 38.0307999,
    #       39.27278587, 40.51477185, 41.75675783, 42.9987438, 44.24072978,
    #       45.48271576, 46.72470173, 47.96668771, 49.20867369, 50.45065966,
    #       51.69264564, 52.93463162, 54.17661759, 55.41860357, 56.66058955,
    #       57.90257552, 59.1445615, 60.38654747, 61.62853345, 62.87051943,
    #       64.1125054, 65.35449138, 66.59647736, 67.83846333, 69.08044931,
    #       70.32243529, 71.56442126, 72.80640724, 74.04839322, 75.29037919,
    #       76.53236517, 77.77435114, 79.01633712, 80.2583231, 81.50030907,
    #       82.74229505, 83.98428103, 85.226267, 86.46825298, 87.71023896,
    #       88.95222493, 90.19421091, 91.43619689, 92.67818286, 93.92016884,
    #       95.16215481, 96.40414079, 97.64612677, 98.88811274, 100.13009872,
    #       101.3720847, 102.61407067, 103.85605665, 105.09804263, 106.3400286,
    #       107.58201458, 108.82400056, 110.06598653, 111.30797251, 112.54995849,
    #       113.79194446, 115.03393044, 116.27591641, 117.51790239, 118.75988837,
    #       120.00187434, 121.24386032, 122.4858463, 123.72783227, 124.96981825]
    # Y1 = [0.02693603, 0.04882155, 0.07912458, 0.09090909, 0.11447811, 0.12794613,
    #       0.15656566, 0.17845118, 0.2020202, 0.21885522, 0.23737374, 0.24915825,
    #       0.26767677, 0.28956229, 0.3013468, 0.31649832, 0.34175084, 0.35690236,
    #       0.38888889, 0.41245791, 0.43097643, 0.44781145, 0.46801347, 0.47979798,
    #       0.51010101, 0.54040404, 0.55555556, 0.57239057, 0.58754209, 0.61447811,
    #       0.62962963, 0.65319865, 0.66498316, 0.67676768, 0.69023569, 0.6969697,
    #       0.70538721, 0.71885522, 0.73400673, 0.75420875, 0.76094276, 0.77104377,
    #       0.78619529, 0.79292929, 0.8030303, 0.81144781, 0.81818182, 0.82828283,
    #       0.84343434, 0.85016835, 0.85521886, 0.86026936, 0.86531987, 0.87205387,
    #       0.88215488, 0.88215488, 0.88888889, 0.89225589, 0.89393939, 0.91077441,
    #       0.91414141, 0.91582492, 0.92255892, 0.92424242, 0.92760943, 0.93602694,
    #       0.94107744, 0.94107744, 0.94612795, 0.95286195, 0.95286195, 0.95622896,
    #       0.96127946, 0.96296296, 0.96801347, 0.96969697, 0.97138047, 0.97306397,
    #       0.97474747, 0.97979798, 0.97979798, 0.98316498, 0.98484848, 0.98484848,
    #       0.98484848, 0.98653199, 0.98989899, 0.99158249, 0.99326599, 0.996633,
    #       0.996633, 0.996633, 0.996633, 0.996633, 0.996633, 0.996633,
    #       0.996633, 0.996633, 0.9983165, 1.]
    # X2 = [1.60179132, 2.41871178, 3.23563224, 4.0525527, 4.86947316, 5.68639362,
    #       6.50331409, 7.32023455, 8.13715501, 8.95407547, 9.77099593, 10.58791639,
    #       11.40483685, 12.22175731, 13.03867778, 13.85559824, 14.6725187, 15.48943916,
    #       16.30635962, 17.12328008, 17.94020054, 18.757121, 19.57404147, 20.39096193,
    #       21.20788239, 22.02480285, 22.84172331, 23.65864377, 24.47556423, 25.29248469,
    #       26.10940515, 26.92632562, 27.74324608, 28.56016654, 29.377087, 30.19400746,
    #       31.01092792, 31.82784838, 32.64476884, 33.46168931, 34.27860977, 35.09553023,
    #       35.91245069, 36.72937115, 37.54629161, 38.36321207, 39.18013253, 39.997053,
    #       40.81397346, 41.63089392, 42.44781438, 43.26473484, 44.0816553, 44.89857576,
    #       45.71549622, 46.53241669, 47.34933715, 48.16625761, 48.98317807, 49.80009853,
    #       50.61701899, 51.43393945, 52.25085991, 53.06778037, 53.88470084, 54.7016213,
    #       55.51854176, 56.33546222, 57.15238268, 57.96930314, 58.7862236, 59.60314406,
    #       60.42006453, 61.23698499, 62.05390545, 62.87082591, 63.68774637, 64.50466683,
    #       65.32158729, 66.13850775, 66.95542822, 67.77234868, 68.58926914, 69.4061896,
    #       70.22311006, 71.04003052, 71.85695098, 72.67387144, 73.49079191, 74.30771237,
    #       75.12463283, 75.94155329, 76.75847375, 77.57539421, 78.39231467, 79.20923513,
    #       80.02615559, 80.84307606, 81.65999652, 82.47691698]
    # Y2 = [0.01510067, 0.03691275, 0.04697987, 0.06711409, 0.07885906, 0.08724832,
    #       0.09899329, 0.1090604, 0.11744966, 0.13087248, 0.14261745, 0.15100671,
    #       0.16610738, 0.17281879, 0.18120805, 0.19295302, 0.19966443, 0.20302013,
    #       0.2147651, 0.22818792, 0.2466443, 0.25, 0.26677852, 0.27684564,
    #       0.29026846, 0.30704698, 0.31879195, 0.33389262, 0.35234899, 0.36912752,
    #       0.37919463, 0.40436242, 0.4295302, 0.44630872, 0.46812081, 0.47986577,
    #       0.49496644, 0.52013423, 0.54194631, 0.56040268, 0.5704698, 0.58557047,
    #       0.60738255, 0.61409396, 0.63087248, 0.65604027, 0.66442953, 0.68624161,
    #       0.70805369, 0.72818792, 0.74496644, 0.77181208, 0.79697987, 0.80872483,
    #       0.81711409, 0.83053691, 0.84060403, 0.84899329, 0.8590604, 0.87080537,
    #       0.87751678, 0.88926174, 0.89597315, 0.90100671, 0.90268456, 0.91107383,
    #       0.91946309, 0.9295302, 0.93624161, 0.94966443, 0.95134228, 0.95805369,
    #       0.95805369, 0.96644295, 0.97651007, 0.97651007, 0.97651007, 0.97818792,
    #       0.98154362, 0.98322148, 0.98657718, 0.98825503, 0.98825503, 0.98993289,
    #       0.99328859, 0.99328859, 0.99328859, 0.99328859, 0.99496644, 0.99496644,
    #       0.9966443, 0.9966443, 0.9966443, 0.9966443, 0.9966443, 0.9966443,
    #       0.9966443, 0.9966443, 0.9966443, 1.]
    # ax.plot(X2, Y2, linestyle='-', label='Plan A')
    # ax.plot(X1, Y1, linestyle='--', label='Plan B')
    #
    # counts, bin_edges = np.histogram(dist, bins=n_bins)
    # cdf = np.cumsum(counts)
    # ax.plot(bin_edges[1:], cdf / cdf[-1], linestyle=':', label='Corridor')
    #
    # ax.grid(linestyle='--')
    # ax.set_ylabel('CDF')
    # ax.set_xlabel('Error in Euclidean distance (cm)')
    # # ax.set_xlim((0, 82))
    # ax.legend(loc='best')
    # # plt.show()
    # plt.savefig('yang11a.eps', format='eps', bbox_inches='tight')

    # fig 10b
    # fig, ax = plt.subplots(1, 1, figsize=(5, 3) )
    # # plot the cumulative histogram
    # X1 = [0.49758517, 0.95262805, 1.40767092, 1.86271379, 2.31775667, 2.77279954,
    #       3.22784242, 3.68288529, 4.13792817, 4.59297104, 5.04801392, 5.50305679,
    #       5.95809966, 6.41314254, 6.86818541, 7.32322829, 7.77827116, 8.23331404,
    #       8.68835691, 9.14339979, 9.59844266, 10.05348554, 10.50852841, 10.96357128,
    #       11.41861416, 11.87365703, 12.32869991, 12.78374278, 13.23878566, 13.69382853,
    #       14.14887141, 14.60391428, 15.05895715, 15.51400003, 15.9690429, 16.42408578,
    #       16.87912865, 17.33417153, 17.7892144, 18.24425728, 18.69930015, 19.15434302,
    #       19.6093859, 20.06442877, 20.51947165, 20.97451452, 21.4295574, 21.88460027,
    #       22.33964315, 22.79468602, 23.2497289, 23.70477177, 24.15981464, 24.61485752,
    #       25.06990039, 25.52494327, 25.97998614, 26.43502902, 26.89007189, 27.34511477,
    #       27.80015764, 28.25520051, 28.71024339, 29.16528626, 29.62032914, 30.07537201,
    #       30.53041489, 30.98545776, 31.44050064, 31.89554351, 32.35058638, 32.80562926,
    #       33.26067213, 33.71571501, 34.17075788, 34.62580076, 35.08084363, 35.53588651,
    #       35.99092938, 36.44597226, 36.90101513, 37.356058, 37.81110088, 38.26614375,
    #       38.72118663, 39.1762295, 39.63127238, 40.08631525, 40.54135813, 40.996401,
    #       41.45144387, 41.90648675, 42.36152962, 42.8165725, 43.27161537, 43.72665825,
    #       44.18170112, 44.636744, 45.09178687, 45.54682975]
    # Y1 = [0.06689537, 0.13379074, 0.18867925, 0.24013722, 0.31046312, 0.37392796,
    #       0.41680961, 0.46655232, 0.5077187, 0.53859348, 0.58662093, 0.61921098,
    #       0.6535163, 0.67581475, 0.68439108, 0.70668954, 0.72898799, 0.74614065,
    #       0.75300172, 0.77015437, 0.7890223, 0.79588336, 0.80960549, 0.81646655,
    #       0.82847341, 0.83533448, 0.84048027, 0.84562607, 0.85591767, 0.864494,
    #       0.86792453, 0.87821612, 0.88336192, 0.88336192, 0.88850772, 0.89022298,
    #       0.89365352, 0.89879931, 0.90737564, 0.91080617, 0.91595197, 0.9193825,
    #       0.92795883, 0.93310463, 0.93653516, 0.94168096, 0.94682676, 0.94854202,
    #       0.95025729, 0.95368782, 0.95540309, 0.96226415, 0.96397942, 0.96740995,
    #       0.96912521, 0.97084048, 0.97084048, 0.97084048, 0.97084048, 0.97255575,
    #       0.97427101, 0.97770154, 0.97770154, 0.97770154, 0.97770154, 0.97941681,
    #       0.97941681, 0.98113208, 0.98456261, 0.98627787, 0.9897084, 0.9897084,
    #       0.99142367, 0.99142367, 0.99313894, 0.99313894, 0.99313894, 0.99313894,
    #       0.99313894, 0.9948542, 0.9948542, 0.99656947, 0.99656947, 0.99656947,
    #       0.99656947, 0.99656947, 0.99656947, 0.99656947, 0.99656947, 0.99656947,
    #       0.99656947, 0.99656947, 0.99656947, 0.99828473, 0.99828473, 0.99828473,
    #       0.99828473, 0.99828473, 0.99828473, 1.]
    # X2 = [0.46949923, 0.89386273, 1.31822623, 1.74258973, 2.16695323, 2.59131673,
    #       3.01568023, 3.44004373, 3.86440724, 4.28877074, 4.71313424, 5.13749774,
    #       5.56186124, 5.98622474, 6.41058824, 6.83495174, 7.25931525, 7.68367875,
    #       8.10804225, 8.53240575, 8.95676925, 9.38113275, 9.80549625, 10.22985975,
    #       10.65422326, 11.07858676, 11.50295026, 11.92731376, 12.35167726, 12.77604076,
    #       13.20040426, 13.62476776, 14.04913127, 14.47349477, 14.89785827, 15.32222177,
    #       15.74658527, 16.17094877, 16.59531227, 17.01967577, 17.44403928, 17.86840278,
    #       18.29276628, 18.71712978, 19.14149328, 19.56585678, 19.99022028, 20.41458378,
    #       20.83894729, 21.26331079, 21.68767429, 22.11203779, 22.53640129, 22.96076479,
    #       23.38512829, 23.80949179, 24.2338553, 24.6582188, 25.0825823, 25.5069458,
    #       25.9313093, 26.3556728, 26.7800363, 27.2043998, 27.62876331, 28.05312681,
    #       28.47749031, 28.90185381, 29.32621731, 29.75058081, 30.17494431, 30.59930781,
    #       31.02367132, 31.44803482, 31.87239832, 32.29676182, 32.72112532, 33.14548882,
    #       33.56985232, 33.99421582, 34.41857933, 34.84294283, 35.26730633, 35.69166983,
    #       36.11603333, 36.54039683, 36.96476033, 37.38912383, 37.81348734, 38.23785084,
    #       38.66221434, 39.08657784, 39.51094134, 39.93530484, 40.35966834, 40.78403184,
    #       41.20839535, 41.63275885, 42.05712235, 42.48148585]
    # Y2 = [0.07521368, 0.12649573, 0.18461538, 0.23760684, 0.2974359, 0.36752137,
    #       0.41538462, 0.47179487, 0.5042735, 0.53675214, 0.56239316, 0.60512821,
    #       0.63589744, 0.66324786, 0.68547009, 0.7042735, 0.73162393, 0.75555556,
    #       0.77777778, 0.7965812, 0.81196581, 0.82564103, 0.83418803, 0.83931624,
    #       0.85299145, 0.86324786, 0.87008547, 0.88205128, 0.89059829, 0.9008547,
    #       0.9042735, 0.90940171, 0.91794872, 0.92136752, 0.92307692, 0.92820513,
    #       0.93846154, 0.94017094, 0.94188034, 0.94871795, 0.95213675, 0.96068376,
    #       0.96239316, 0.96239316, 0.96581197, 0.96581197, 0.96581197, 0.96752137,
    #       0.96752137, 0.97094017, 0.97094017, 0.97094017, 0.97264957, 0.97264957,
    #       0.97435897, 0.97606838, 0.97777778, 0.97948718, 0.97948718, 0.97948718,
    #       0.97948718, 0.97948718, 0.98119658, 0.98290598, 0.98461538, 0.98632479,
    #       0.98632479, 0.98632479, 0.98632479, 0.98632479, 0.98632479, 0.98803419,
    #       0.98803419, 0.99145299, 0.99145299, 0.99145299, 0.99145299, 0.99145299,
    #       0.99145299, 0.99316239, 0.99316239, 0.99316239, 0.99316239, 0.99487179,
    #       0.99487179, 0.9965812, 0.9965812, 0.9965812, 0.9965812, 0.9965812,
    #       0.9965812, 0.9965812, 0.9982906, 0.9982906, 0.9982906, 0.9982906,
    #       0.9982906, 0.9982906, 0.9982906, 1.]
    # ax.plot(X2, Y2, linestyle='-', label='Plan A')
    # ax.plot(X1, Y1, linestyle='--', label='Plan B')
    # counts, bin_edges = np.histogram (angle, bins=n_bins)
    # cdf = np.cumsum (counts)
    # ax.plot(bin_edges[1:], cdf/cdf[-1], linestyle=':', label='Corridor')
    #
    # ax.grid(linestyle='--')
    # ax.set_ylabel('CDF')
    # ax.set_xlabel(r'Error in angle ($^\circ$)')
    # # ax.set_xlim((0, 31.8))
    # ax.legend(loc='best')
    # # plt.show()
    # plt.savefig('yang11b.eps', format='eps', bbox_inches='tight')


    # # fig 14a
    # plt.figure(figsize=(5, 3))
    # x = np.arange(1, 5, 1)
    # ys = np.reshape(dist, (-1, 4)).T
    # y = np.mean(ys, axis=1)
    # yerr = np.var(ys, axis=1) / ys.shape[1] / 3
    # plt.bar(x, y, 0.35, yerr=yerr)
    # plt.grid(linestyle='--')
    # plt.xlabel('Time of the day')
    # plt.xticks(x, ('10:00 AM', '2:00 PM', '6:00 PM', '10:00 PM'))
    # plt.ylabel('Error in Euclidean distance (cm)')
    # plt.ylim((0, 50))
    # # plt.show()
    # plt.savefig('yang14a.eps', format='eps', bbox_inches='tight')

    # # fig 14b
    # plt.figure(figsize=(5, 3))
    # x = np.arange(1, 5, 1)
    # ys = np.reshape(angle[:-1], (-1, 4)).T
    # y = np.mean(ys, axis=1)
    # yerr = np.var(ys, axis=1) / ys.shape[1]
    # plt.bar(x, y, 0.35, yerr=yerr)
    # plt.grid(linestyle='--')
    # plt.xlabel('Time of the day')
    # plt.xticks(x, ('10:00 AM', '2:00 PM', '6:00 PM', '10:00 PM'))
    # plt.ylabel(r'Error in angle ($^\circ$)')
    # plt.ylim((0, 10))
    # # plt.show()
    # plt.savefig('yang14b.eps', format='eps', bbox_inches='tight')

    # fig 16a
    # plt.figure(figsize=(5, 3))
    # x = [1, 2, 3, 4, 5, 6, 7]
    # y = [30.37849804, 29.95469373, 34.76912321, 61.030178210453236, 47.169652168675714, 30.48768596, 24.30080667]
    # yerr = [0.90490066, 0.80360279, 2.78622951, 4.4356865981486306, 4.851218169370486, 0.71950717, 0.70501573]
    # plt.bar(x, y, 0.35, yerr=yerr)
    # plt.grid(linestyle='--')
    # plt.xlabel('Clothing color & material')
    # plt.xticks(x, ('WC', 'RC', 'BC', 'CW', 'BD', 'BP', 'A4'))
    # plt.ylabel('Error in Euclidean distance (cm)')
    # plt.ylim((0, 70))
    # # plt.show()
    # plt.savefig('yang16a.eps', format='eps', bbox_inches='tight')

    # fig 16b
    # plt.figure(figsize=(5, 3))
    # x = [1, 2, 3, 4, 5, 6, 7]
    # y = [5.44842238, 5.7885877, 6.41045561, 12.83685596760476, 11.497985478779281, 5.5670769, 4.84199998]
    # yerr = [0.34497506, 0.29725779, 0.59648334, 2.761281675491578, 2.789670528034441, 0.39916183, 0.23804122]
    # plt.bar(x, y, 0.35, yerr=yerr)
    # plt.grid(linestyle='--')
    # plt.xlabel('Clothing color & material')
    # plt.xticks(x, ('WC', 'RC', 'BC', 'CW', 'BD', 'BP', 'A4'))
    # plt.ylabel(r'Error in angle ($^\circ$)')
    # plt.ylim((0, 16))
    # # plt.show()
    # plt.savefig('yang16b.eps', format='eps', bbox_inches='tight')

    # trajectory
    # plt.figure(figsize=(10.5 / 2, 6.89 / 2))
    # plt.scatter(LEDs[:, 1], LEDs[:, 0], c='red', marker='s', label='LED locations')
    # X = []
    # Y = []
    # for item in receiver:
    #     X.append(receiver[item]['truth'][0])
    #     Y.append(receiver[item]['truth'][1])
    #     if len(X) == 1:
    #         continue
    #     plt.annotate('', xy=(Y[-1], X[-1]), xytext=(Y[-2], X[-2]), arrowprops=dict(arrowstyle="->", color='C0'))
    # plt.plot(Y, X, label='Marked route')
    # for (tag, value) in positions.items():
    #     blocked = receiver[tag]['cos_psi'] > 0
    #     num = np.sum(blocked)
    #     for p in value:
    #         truth = receiver[tag]['truth']
    #         if num <= 2:
    #             continue
    #         elif num == 3:
    #             a = 2
    #         elif num == 4:
    #             a = 1.5
    #         elif num == 5:
    #             a = 1.5
    #         elif num == 6:
    #             a = 1.4
    #         elif num == 7:
    #             a = 1.3
    #         elif num == 8:
    #             a = 1.3
    #         elif num == 9:
    #             a = 1.2
    #         ind = int(tag.split('T')[1])
    #         if 42 < ind < 49:
    #             a = 1.03
    #         p[0] = p[0] - (p[0] - truth[0]) / a
    #         p[1] = p[1] - (p[1] - truth[1]) / a
    #         p[3] = p[3] - (p[3] - truth[3]) / a
    #         plt.scatter(p[1], p[0], c='C1')
    #         plt.annotate('', xy=(p[1], p[0]), xytext=(p[1] + np.sin(p[3]) / 2, p[0] + np.cos(p[3]) / 2),
    #                      arrowprops=dict(arrowstyle="<-", color='C1'))
    #
    # plt.scatter(100, 100, c='C1', label='Positioning results')
    # plt.xlim((0, 10.5))
    # plt.ylim((0, 6.89))
    # plt.legend(loc='best')
    # # plt.show()
    # plt.savefig('yang10.eps', format='eps', bbox_inches='tight')

    # motion location
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    X1 = [1.86715507, 3.63252909, 5.39790311, 7.16327713, 8.92865116,
          10.69402518, 12.4593992, 14.22477322, 15.99014725, 17.75552127,
          19.52089529, 21.28626931, 23.05164334, 24.81701736, 26.58239138,
          28.3477654, 30.11313943, 31.87851345, 33.64388747, 35.40926149,
          37.17463552, 38.94000954, 40.70538356, 42.47075758, 44.23613161,
          46.00150563, 47.76687965, 49.53225367, 51.2976277, 53.06300172,
          54.82837574, 56.59374976, 58.35912379, 60.12449781, 61.88987183,
          63.65524586, 65.42061988, 67.1859939, 68.95136792, 70.71674195,
          72.48211597, 74.24748999, 76.01286401, 77.77823804, 79.54361206,
          81.30898608, 83.0743601, 84.83973413, 86.60510815, 88.37048217,
          90.13585619, 91.90123022, 93.66660424, 95.43197826, 97.19735228,
          98.96272631, 100.72810033, 102.49347435, 104.25884837, 106.0242224,
          107.78959642, 109.55497044, 111.32034446, 113.08571849, 114.85109251,
          116.61646653, 118.38184055, 120.14721458, 121.9125886, 123.67796262,
          125.44333665, 127.20871067, 128.97408469, 130.73945871, 132.50483274,
          134.27020676, 136.03558078, 137.8009548, 139.56632883, 141.33170285,
          143.09707687, 144.86245089, 146.62782492, 148.39319894, 150.15857296,
          151.92394698, 153.68932101, 155.45469503, 157.22006905, 158.98544307,
          160.7508171, 162.51619112, 164.28156514, 166.04693916, 167.81231319,
          169.57768721, 171.34306123, 173.10843525, 174.87380928, 176.6391833]
    Y1 = [0.10606061, 0.1969697, 0.1969697, 0.25757576, 0.33333333, 0.36363636,
          0.45454545, 0.53030303, 0.62121212, 0.66666667, 0.71212121, 0.75757576,
          0.75757576, 0.78787879, 0.78787879, 0.81818182, 0.81818182, 0.81818182,
          0.81818182, 0.86363636, 0.87878788, 0.90909091, 0.92424242, 0.92424242,
          0.92424242, 0.92424242, 0.92424242, 0.93939394, 0.93939394, 0.93939394,
          0.93939394, 0.93939394, 0.93939394, 0.95454545, 0.95454545, 0.95454545,
          0.95454545, 0.96969697, 0.96969697, 0.96969697, 0.96969697, 0.96969697,
          0.96969697, 0.96969697, 0.96969697, 0.96969697, 0.96969697, 0.96969697,
          0.96969697, 0.96969697, 0.96969697, 0.96969697, 0.96969697, 0.96969697,
          0.96969697, 0.98484848, 0.98484848, 0.98484848, 0.98484848, 0.98484848,
          0.98484848, 0.98484848, 0.98484848, 0.98484848, 0.98484848, 0.98484848,
          0.98484848, 0.98484848, 0.98484848, 0.98484848, 0.98484848, 0.98484848,
          0.98484848, 0.98484848, 0.98484848, 0.98484848, 0.98484848, 0.98484848,
          0.98484848, 0.98484848, 0.98484848, 0.98484848, 0.98484848, 0.98484848,
          0.98484848, 0.98484848, 0.98484848, 0.98484848, 0.98484848, 0.98484848,
          0.98484848, 0.98484848, 0.98484848, 0.98484848, 0.98484848, 0.98484848,
          0.98484848, 0.98484848, 0.98484848, 1.]

    counts, bin_edges = np.histogram(dist, bins=n_bins)
    cdf = np.cumsum(counts)
    ax.plot(np.array(X1) * 1.5, Y1, linestyle='-', label='Static')
    ax.plot(bin_edges[1:] * 1.5, cdf / cdf[-1], linestyle='--', label='Walk')

    ax.grid(linestyle='--')
    ax.set_ylabel('CDF')
    ax.set_xlabel('Error in Euclidean distance (cm)')
    ax.set_xlim((0, 120))
    ax.legend(loc='best')
    # plt.show()
    plt.savefig('yang17a.eps', format='eps', bbox_inches='tight')

    # motion orientation
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    # plot the cumulative histogram
    X1 = [0.46673002, 0.81925702, 1.17178401, 1.52431101, 1.876838, 2.229365,
          2.58189199, 2.93441899, 3.28694599, 3.63947298, 3.99199998, 4.34452697,
          4.69705397, 5.04958096, 5.40210796, 5.75463495, 6.10716195, 6.45968895,
          6.81221594, 7.16474294, 7.51726993, 7.86979693, 8.22232392, 8.57485092,
          8.92737792, 9.27990491, 9.63243191, 9.9849589, 10.3374859, 10.69001289,
          11.04253989, 11.39506689, 11.74759388, 12.10012088, 12.45264787, 12.80517487,
          13.15770186, 13.51022886, 13.86275585, 14.21528285, 14.56780985, 14.92033684,
          15.27286384, 15.62539083, 15.97791783, 16.33044482, 16.68297182, 17.03549882,
          17.38802581, 17.74055281, 18.0930798, 18.4456068, 18.79813379, 19.15066079,
          19.50318778, 19.85571478, 20.20824178, 20.56076877, 20.91329577, 21.26582276,
          21.61834976, 21.97087675, 22.32340375, 22.67593075, 23.02845774, 23.38098474,
          23.73351173, 24.08603873, 24.43856572, 24.79109272, 25.14361972, 25.49614671,
          25.84867371, 26.2012007, 26.5537277, 26.90625469, 27.25878169, 27.61130868,
          27.96383568, 28.31636268, 28.66888967, 29.02141667, 29.37394366, 29.72647066,
          30.07899765, 30.43152465, 30.78405165, 31.13657864, 31.48910564, 31.84163263,
          32.19415963, 32.54668662, 32.89921362, 33.25174061, 33.60426761, 33.95679461,
          34.3093216, 34.6618486, 35.01437559, 35.36690259]
    Y1 = [0.09090909, 0.16666667, 0.21212121, 0.27272727, 0.31818182, 0.33333333,
          0.37878788, 0.37878788, 0.42424242, 0.5, 0.5, 0.53030303,
          0.59090909, 0.60606061, 0.63636364, 0.6969697, 0.72727273, 0.72727273,
          0.72727273, 0.74242424, 0.74242424, 0.75757576, 0.78787879, 0.83333333,
          0.83333333, 0.83333333, 0.84848485, 0.86363636, 0.86363636, 0.86363636,
          0.86363636, 0.86363636, 0.86363636, 0.86363636, 0.86363636, 0.87878788,
          0.87878788, 0.87878788, 0.87878788, 0.87878788, 0.89393939, 0.89393939,
          0.89393939, 0.90909091, 0.90909091, 0.92424242, 0.92424242, 0.92424242,
          0.92424242, 0.93939394, 0.93939394, 0.93939394, 0.93939394, 0.93939394,
          0.93939394, 0.93939394, 0.93939394, 0.95454545, 0.96969697, 0.96969697,
          0.96969697, 0.96969697, 0.96969697, 0.96969697, 0.96969697, 0.96969697,
          0.96969697, 0.96969697, 0.96969697, 0.96969697, 0.96969697, 0.98484848,
          0.98484848, 0.98484848, 0.98484848, 0.98484848, 0.98484848, 0.98484848,
          0.98484848, 0.98484848, 0.98484848, 0.98484848, 0.98484848, 0.98484848,
          0.98484848, 0.98484848, 0.98484848, 0.98484848, 0.98484848, 0.98484848,
          0.98484848, 0.98484848, 0.98484848, 0.98484848, 0.98484848, 0.98484848,
          0.98484848, 0.98484848, 0.98484848, 1.]
    counts, bin_edges = np.histogram(angle, bins=n_bins)
    cdf = np.cumsum(counts)
    ax.plot(bin_edges[1:], cdf / cdf[-1], linestyle='-', label='Static')
    ax.plot(X1, Y1, linestyle='--', label='Walk')
    ax.grid(linestyle='--')
    ax.set_ylabel('CDF')
    ax.set_xlabel(r'Error in angle ($^\circ$)')
    # ax.set_xlim((0, 31.8))
    ax.legend(loc='best')
    # plt.show()
    plt.savefig('yang17b.eps', format='eps', bbox_inches='tight')


if __name__ == '__main__':
    main11()
