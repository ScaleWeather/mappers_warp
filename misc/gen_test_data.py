import numpy


def main():
    rng = numpy.random.default_rng(seed=42)

    arr = rng.random((33, 33))

    numpy.save("./test-data/random_square_33.npy", arr)


if __name__ == "__main__":
    main()
