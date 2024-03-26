import numpy
from matplotlib import pyplot as plt


def main():
    source = numpy.load("./test-data/waves_34.npy")
    target = numpy.load("./test-data/result.npy")

    ixs = numpy.load("./test-data/ixs.npy")
    jys = numpy.load("./test-data/jys.npy")

    anchors_x = numpy.load("./test-data/dbg_anchors_x.npy")
    anchors_y = numpy.load("./test-data/dbg_anchors_y.npy")

    fig, ax = plt.subplots(3, 2)

    ax[0][0].imshow(source)
    ax[0][1].imshow(target)

    cr1 = ax[1][0].imshow(ixs)
    cr2 = ax[1][1].imshow(jys)

    cr3 = ax[2][0].imshow(anchors_x)
    cr4 = ax[2][1].imshow(anchors_y)

    fig.colorbar(cr1, ax=ax[1][0])
    fig.colorbar(cr2, ax=ax[1][1])

    fig.colorbar(cr3, ax=ax[2][0])
    fig.colorbar(cr4, ax=ax[2][1])

    plt.show()


if __name__ == "__main__":
    main()
