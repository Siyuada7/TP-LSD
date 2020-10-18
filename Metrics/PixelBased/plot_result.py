import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.io as sio
import collections


def main():

    exps = ['path/result.mat']
    labels = ['method']

    print(exps, labels)

    num = 0
    for exp, label in zip(exps, labels):
        mat = sio.loadmat(exp)

        prs = mat['sumprecisions']
        rcs = mat['sumrecalls']
        N = mat['nsamples']
        prs = prs[:, 0]
        rcs = rcs[:, 0]
        N = N[:, 0]
        prs = [x / float(n) for x, n in zip(prs, N)]
        rcs = [x / float(n) for x, n in zip(rcs, N)]
        F = 2 * np.array(prs) * np.array(rcs) / (np.array(prs) + np.array(rcs))
        index = F < 1
        index1 = F > 0.1
        rrcs, pprs = [], []
        for k in range(len(index)):
            if index[k] and index1[k]:
                rrcs.append(rcs[k])
                pprs.append(prs[k])
        line_x = plt.plot(rrcs, pprs, '-*', linewidth=2, label=label)  # color=color[num]
        num += 1

        print(
            label, "measure is: ",
            (2 * np.array(pprs) * np.array(rrcs) / (np.array(pprs) + np.array(rrcs))).max(),
            'pos: ', np.argmax(2 * np.array(pprs) * np.array(rrcs) / (np.array(pprs) + np.array(rrcs))) + 1
        )
    f_scores = np.linspace(0.2, 0.8, num=8)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color="green", alpha=0.3)
        plt.annotate("f={0:0.1}".format(f_score), xy=(0.9, y[45] + 0.02), alpha=0.4)
    plt.grid(True)

    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.xticks(np.arange(0, 1.0, step=0.1))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.yticks(np.arange(0.0, 1.0, step=0.1))
    plt.legend(loc=3)
    plt.title("PR Curve for Heatmap in Wireframe dataset")
    plt.show()


if __name__ == '__main__':
    main()
