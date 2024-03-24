import numpy
import rasterio


def main():
    ll = rasterio.open("./misc/lonlat-p25.tif").read()
    lcc = rasterio.open("./misc/lcc-india-p25.tif").read()

    numpy.save("./misc/ll-p25.npy", ll)
    numpy.save("./misc/lcc-p25.npy", lcc)

if __name__ == '__main__':
    main()