from matplotlib import pyplot as plt
import numpy


def main():
    rng = numpy.random.default_rng(seed=42)

    arr = rng.random((34, 34))

    numpy.save("./test-data/random_square_34.npy", arr)

    x = numpy.linspace(1, 5, 34)
    y = numpy.linspace(1, 7, 34)

    grd_x, grd_y = numpy.meshgrid(x, y)

    z = numpy.sin(3*grd_x)+numpy.sin(grd_x*grd_y)*numpy.cos(2*grd_y)

    plt.imshow(z)
    plt.show()

    numpy.save("./test-data/waves_34.npy", z)


if __name__ == "__main__":
    main()
