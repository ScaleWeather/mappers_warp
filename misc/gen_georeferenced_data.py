import numpy
import cfgrib


def main():
    x = numpy.linspace(1, 5, 34)
    y = numpy.linspace(1, 7, 34)

    grd_x, grd_y = numpy.meshgrid(x, y)

    vals = numpy.sin(3*grd_x)+numpy.sin(grd_x*grd_y)*numpy.cos(2*grd_y)

    gfs = cfgrib.open_datasets("./misc/gfs-p25.grib", backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})[0]

    ix = 240
    iy = 200

    data = gfs["t2m"][iy:iy+34, ix:ix+34] 
    vals = vals.reshape(data.shape)
    data.values = vals

    print(data)

    data.rio.write_crs("EPSG:4326", inplace=True)

    data.rio.to_raster("./misc/lonlat-waves.tif")


if __name__ == "__main__":
    main()
