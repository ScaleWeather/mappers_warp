import numpy
import rasterio


def main():
    ll = rasterio.open("./misc/lcc-waves.tif").read().squeeze()

    ll = numpy.array(ll)

    ll = ll.astype(numpy.float64)

    numpy.save("./test-data/waves_ref.npy", ll)

if __name__ == '__main__':
    main()