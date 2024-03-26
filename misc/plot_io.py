import numpy
from matplotlib import pyplot as plt


def main():
    ref = numpy.load("./test-data/waves_ref.npy")
    tgt = numpy.load("./misc/waves_34_warped.npy")

    diff = ref - tgt

    dx = numpy.diff(tgt, axis=1)
    dy = numpy.diff(tgt, axis=0)

    print(dx.max(), dx.min())
    print(dy.max(), dy.min())

    fig, ax = plt.subplots(1, 3)

    cr1 = ax[0].imshow(ref)
    cr2 = ax[1].imshow(diff)
    cr3 = ax[2].imshow(tgt)

    fig.colorbar(cr1, ax=ax[0])
    fig.colorbar(cr2, ax=ax[1])
    fig.colorbar(cr3, ax=ax[2])

    plt.show()


if __name__ == "__main__":
    main()
