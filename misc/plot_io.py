import numpy
from matplotlib import pyplot as plt


def main():
    source = numpy.load("./test-data/waves_34.npy")
    target = numpy.load("./test-data/result.npy")

    fig, ax = plt.subplots(1, 2)

    cr1 = ax[0].imshow(source)
    cr2 = ax[1].imshow(target)

    fig.colorbar(cr1, ax=ax[0])
    fig.colorbar(cr2, ax=ax[1])

    plt.show()


if __name__ == "__main__":
    main()
