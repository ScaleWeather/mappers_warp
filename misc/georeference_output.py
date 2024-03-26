import numpy
import rasterio


def main():
    ref = rasterio.open("./misc/lcc-waves.tif")
    tgt = numpy.load("./misc/waves_34_warped.npy")

    new_dataset = rasterio.open(
        "./misc/warped.tif",
        "w",
        driver="GTiff",
        height=tgt.shape[0],
        width=tgt.shape[1],
        count=1,
        dtype=tgt.dtype,
        crs=ref.crs,
        transform=ref.transform,
    )

    new_dataset.write(tgt, 1)


if __name__ == "__main__":
    main()
