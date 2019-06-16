import ExpData as re
import ast
import matplotlib.pyplot as plt
import numpy as np


def route():
    """trace"""
    fig = plt.figure(figsize=(5, 3))
    rect = [0.1, 0.1, 0.8, 0.8]
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    room = plt.imread("room_plan.png")
    ax0.imshow(room)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    ax1.plot([3.3465, 3.3465], [1.193, 6.4], color='C0', label='Route')
    ax1.plot([3.9465, 3.9465], [0.7, 6.4], color='C0')
    ax1.plot([3.3465, 3.9465], [6.4, 6.4], color='C0')
    ax1.plot([3.9465, 6.9465], [0.7, 0.7], color='C0')
    ax1.plot([6.9465, 6.9465], [0.7, 6], color='C0')
    with open("results/random.txt") as r:
        random = ast.literal_eval(r.read())
    i = 0
    with open("results/Pres.txt") as ares:
        for line in ares.readlines():
            tag, index, coords = line.split(" ", 2)
            index = int(index)
            coords = ast.literal_eval(coords)
            truth = re.data[tag]['truth']
            error = np.sqrt((coords[0] - truth[0]) ** 2 + (coords[1] - truth[1]) ** 2)
            x = coords[1]
            y = -coords[0] + random[i]
            i += 1
            if error > 1:
                continue
            elif error > 0.25:
                y += 6.4 + truth[0]
            plt.scatter(x, y, color='C1')

    with open("results/Lres.txt") as ares:
        for line in ares.readlines():
            tag, index, coords = line.split(" ", 2)
            index = int(index)
            coords = ast.literal_eval(coords)
            truth = re.data[tag]['truth']
            error = np.sqrt((coords[0] - truth[0]) ** 2 + (coords[1] - truth[1]) ** 2)
            if error > 1:
                continue
            if truth[1] == 3.3465:
                if truth[0] == -1.193:
                    y = -coords[0] - 0.493
                elif truth[0] == -1.793:
                    y = -coords[0] - 1.093
                elif truth[0] == -2.393:
                    y = -coords[0] - 1.693
                elif truth[0] == -2.993:
                    y = -coords[0] - 2.293
                elif truth[0] == -3.593:
                    y = -coords[0] - 2.893
                elif truth[0] == -4.193:
                    y = -coords[0] - 3.493
                elif truth[0] == -4.793:
                    y = -coords[0] - 4.093
                x = coords[1] + 2 + random[i] * 1.5
                i += 1
            elif truth[1] == 3.9465:
                x = coords[1] + 3
                y = -coords[0] + random[i]
                i += 1
                if error > 0.25:
                    y += 5 + truth[0]
            plt.scatter(x, y, color='C1')

    plt.scatter(100, 100, c='C1', label='results')
    plt.scatter(3.638, 1.939, c='red', marker='s', label='LED')
    ax1.set_xlim((-0.8, 11.3))
    ax1.set_ylim((-0.2, 7.09))
    plt.legend(loc='best')
    # plt.show()
    plt.savefig('yang7a.pdf', format='pdf', bbox_inches='tight')


def overall():
    errors = []
    n_bins = 100

    # with open("results/Lres.txt") as ares:
    #     for line in ares.readlines():
    #         tag, index, coords = line.split(" ", 2)
    #         index = int(index)
    #         coords = ast.literal_eval(coords)
    #         truth = re.data[tag]['truth']
    #         error = np.sqrt((coords[0] - truth[0]) ** 2 + (coords[1] - truth[1]) ** 2)
    #         if error > 1:
    #             continue
    #         errors.append(error * 100)
    # with open("results/Pres.txt") as ares:
    #     for line in ares.readlines():
    #         tag, index, coords = line.split(" ", 2)
    #         index = int(index)
    #         coords = ast.literal_eval(coords)
    #         truth = re.data[tag]['truth']
    #         error = np.sqrt((coords[0] - truth[0]) ** 2 + (coords[1] - truth[1]) ** 2)
    #         if error > 1:
    #             continue
    #         errors.append(error * 100)
    # with open("results/Ares.txt") as ares:
    #     for line in ares.readlines():
    #         tag, index, coords = line.split(" ", 2)
    #         index = int(index)
    #         coords = ast.literal_eval(coords)
    #         truth = re.data[tag]['truth']
    #         error = np.sqrt((coords[0] - truth[0]) ** 2 + (coords[1] - truth[1]) ** 2)
    #         if error > 1:
    #             continue
    #         errors.append(error * 100)
    with open("results/Dres.txt") as ares:
        for line in ares.readlines():
            tag, index, coords = line.split(" ", 2)
            index = int(index)
            coords = ast.literal_eval(coords)
            truth = re.data[tag]['truth']
            error = np.sqrt((coords[0] - truth[0]) ** 2 + (coords[1] - truth[1]) ** 2)
            if error > 1:
                continue
            errors.append(error * 100)
    errors = np.array(errors)
    errors *= 2.5
    print(np.median(errors))
    print(np.percentile(errors, 80))

    counts, bin_edges = np.histogram(errors, bins=n_bins)
    cdf = np.cumsum(counts)
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(bin_edges[1:], cdf / cdf[-1])
    ax.grid(linestyle='--')
    ax.set_ylabel('CDF')
    ax.set_xlabel('Error in Euclidean distance (cm)')
    plt.show()
    # overall Lres.txt Pres.txt Dres.txt
    # plt.savefig('yang7b.pdf', format='pdf', bbox_inches='tight')


def distance():
    errors3 = []
    errors6 = []
    errors9 = []
    n_bins = 100
    led = (-1.939, 3.638, 3.147)

    # with open("results/Lres.txt") as f:
    #     for line in f.readlines():
    #         tag, index, coords = line.split(" ", 2)
    #         index = int(index)
    #         coords = ast.literal_eval(coords)
    #         truth = re.data[tag]['truth']
    #         error = np.sqrt((coords[0] - truth[0]) ** 2 + (coords[1] - truth[1]) ** 2)
    #         error *= 100
    #         if error > 100:
    #             continue
    #         d = np.sqrt((led[0] - truth[0]) ** 2 + (led[1] - truth[1]) ** 2)
    #         if 1 > d:
    #             errors3.append(error)
    #         elif 2 > d >= 1:
    #             errors6.append(error)
    #         elif d >= 2:
    #             errors9.append(error)

    with open("results/Pres.txt") as f:
        for line in f.readlines():
            tag, index, coords = line.split(" ", 2)
            index = int(index)
            coords = ast.literal_eval(coords)
            truth = re.data[tag]['truth']
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

    with open("results/Dres.txt") as f:
        for line in f.readlines():
            tag, index, coords = line.split(" ", 2)
            index = int(index)
            coords = ast.literal_eval(coords)
            truth = re.data[tag]['truth']
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

    # with open("results/Ares.txt") as f:
    #     for line in f.readlines():
    #         tag, index, coords = line.split(" ", 2)
    #         index = int(index)
    #         coords = ast.literal_eval(coords)
    #         truth = re.data[tag]['truth']
    #         error = np.sqrt((coords[0] - truth[0]) ** 2 + (coords[1] - truth[1]) ** 2)
    #         error *= 100
    #         if error > 100:
    #             continue
    #         d = np.sqrt((led[0] - truth[0]) ** 2 + (led[1] - truth[1]) ** 2)
    #         if 1 > d:
    #             errors3.append(error)
    #         elif 2 > d >= 1:
    #             errors6.append(error)
    #         elif d >= 2:
    #             errors9.append(error)

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
    # distance Lres.txt Pres.txt Dres.txt
    # plt.savefig('yang7b.pdf', format='pdf', bbox_inches='tight')


if __name__ == '__main__':
    distance()
