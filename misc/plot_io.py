import numpy
from matplotlib import pyplot as plt


def main():
    spline = numpy.load("./tests/data/gfs_t2m_cubic.npy")
    mitchell = numpy.load("./tests/data/gfs_t2m_mitchell.npy")

    diff = spline - mitchell

    fig, ax = plt.subplots(1, 3)

    cr1 = ax[0].imshow(spline)
    cr2 = ax[1].imshow(mitchell)
    cr3 = ax[2].imshow(diff)

    fig.colorbar(cr1, ax=ax[0])
    fig.colorbar(cr2, ax=ax[1])
    fig.colorbar(cr3, ax=ax[2])

    plt.show()


if __name__ == "__main__":
    main()
