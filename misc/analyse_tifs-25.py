import numpy
import pyproj
import rasterio

def padfTransform(lcc_i, lcc_j, rev_transformer, ll_x0, ll_y0):
    # corners of the raster
    xmin = 2320000
    ymax = 5640000

    dx = 10000
    dy = 10000

    # shift here is to get xy value of the center of the pixel
    x, y = lcc_i + 0.5, lcc_j + 0.5
    lcc_x = xmin + x * dx
    lcc_y = ymax - y * dy
    lcc_lonlat = rev_transformer.transform(lcc_x, lcc_y)

    # shifts are a result of using lonlat vaues in ll as midpoints
    ll_x0 = ll_x0 - (0.25/2)
    ll_y0 = ll_y0 + (0.25/2)

    # scaling is 1/dx, 1/dy
    padfX, padfY = (lcc_lonlat[0] - ll_x0)*4, (ll_y0 - lcc_lonlat[1]) * 4

    return padfX, padfY



def find_nearest_index(array, value):
    array = numpy.asarray(array)
    idx = (numpy.abs(array - value)).argmin()

    idx = numpy.unravel_index(idx, array.shape)

    return idx


# values
# 31	19.5	16.7000007629395	68.8000030517578
# 66.9000015258789	81.9000015258789	57.0999984741211	89.4000015258789
# 72.5	34.2000007629395	72.5999984741211	1.20000004768372
# 73.9000015258789	31.2000007629395	38.9000015258789	43.4000015258789

# result: 59.4272662886343

# const double dfSrcXSizeRaw = std::max(
#     0.0, std::min(static_cast<double>(nRasterXSize - nMinXOutClamped),
#                   dfMaxXOut - dfMinXOut));
# const double dfSrcYSizeRaw = std::max(
#     0.0, std::min(static_cast<double>(nRasterYSize - nMinYOutClamped),
#                   dfMaxYOut - dfMinYOut));

# dfXScale = static_cast<double>(nDstXSize) / (nSrcXSize - dfSrcXExtraSize);
# dfYScale = static_cast<double>(nDstYSize) / (nSrcYSize - dfSrcYExtraSize);
# nDst, nSrc are grid sizes (after clamping)

# *pdfSrcXExtraSize = *pnSrcXSize - dfSrcXSizeRaw;
# *pdfSrcYExtraSize = *pnSrcYSize - dfSrcYSizeRaw;


# const int nMinXOutClamped = static_cast<int>(std::max(0.0, dfMinXOut));
# const int nMinYOutClamped = static_cast<int>(std::max(0.0, dfMinYOut));
# const int nMaxXOutClamped = static_cast<int>(
#     std::min(ceil(dfMaxXOut), static_cast<double>(nRasterXSize)));
# const int nMaxYOutClamped = static_cast<int>(
#     std::min(ceil(dfMaxYOut), static_cast<double>(nRasterYSize)));

# *pnSrcXOff =
#     std::max(0, std::min(nMinXOutClamped - nXRadius, nRasterXSize));
# *pnSrcXSize =
#     std::max(0, std::min(nRasterXSize - *pnSrcXOff,
#                          nMaxXOutClamped - *pnSrcXOff + nXRadius));

# *pnSrcYOff =
#     std::max(0, std::min(nMinYOutClamped - nYRadius, nRasterYSize));
# *pnSrcYSize =
#     std::max(0, std::min(nRasterYSize - *pnSrcYOff,
#                          nMaxYOutClamped - *pnSrcYOff + nYRadius));



# dfMinXOut = {double} 4.7102160316373727
# dfMinYOut = {double} 7.1311345893376767
# dfMaxXOut = {double} 25.909471574464362
# dfMaxYOut = {double} 28.519130649598168

# nMinXOutClamped = {const int} 4
# nMinYOutClamped = {const int} 7
# nMaxXOutClamped = {const int} 26
# nMaxYOutClamped = {const int} 29


def get_ll_bounds(ll_raster):
    # corners of the raster
    ul = ll_raster.xy(0, 0)
    ur = ll_raster.xy(ll_raster.width, 0)
    lr = ll_raster.xy(ll_raster.width, ll_raster.height)
    ll = ll_raster.xy(0, ll_raster.height)

    ul_lonlat = ul
    ur_lonlat = ur
    lr_lonlat = lr
    ll_lonlat = ll

    xMin = numpy.min([ul_lonlat[0], ur_lonlat[0], lr_lonlat[0], ll_lonlat[0]])
    xMax = numpy.max([ul_lonlat[0], ur_lonlat[0], lr_lonlat[0], ll_lonlat[0]])
    yMin = numpy.min([ul_lonlat[1], ur_lonlat[1], lr_lonlat[1], ll_lonlat[1]])
    yMax = numpy.max([ul_lonlat[1], ur_lonlat[1], lr_lonlat[1], ll_lonlat[1]])

    return xMin, yMax, xMax, yMin


def get_lcc_bounds(lcc_raster, rev_transformer):
    # corners of the raster
    xmin = 2320000
    ymin = 5090000
    xmax = 2740000
    ymax = 5640000

    ul_lonlat = rev_transformer.transform(xmin, ymax)
    ur_lonlat = rev_transformer.transform(xmax, ymax)
    lr_lonlat = rev_transformer.transform(xmax, ymin)
    ll_lonlat = rev_transformer.transform(xmin, ymin)

    xMin = numpy.min([ul_lonlat[0], ur_lonlat[0], lr_lonlat[0], ll_lonlat[0]])
    xMax = numpy.max([ul_lonlat[0], ur_lonlat[0], lr_lonlat[0], ll_lonlat[0]])
    yMin = numpy.min([ul_lonlat[1], ur_lonlat[1], lr_lonlat[1], ll_lonlat[1]])
    yMax = numpy.max([ul_lonlat[1], ur_lonlat[1], lr_lonlat[1], ll_lonlat[1]])

    return xMin, yMax, xMax, yMin


def main():
    ll = rasterio.open("./misc/lonlat-p25.tif")
    lcc = rasterio.open("./misc/lcc-india-p25.tif")

    transformer = pyproj.Transformer.from_crs(ll.crs, lcc.crs, always_xy=True)
    rev_transformer = pyproj.Transformer.from_crs(lcc.crs, ll.crs, always_xy=True)

    src_values = numpy.array(
        [
            [31, 19.5, 16.7000007629395, 68.8000030517578],
            [66.9000015258789, 81.9000015258789, 57.0999984741211, 89.4000015258789],
            [72.5, 34.2000007629395, 72.5999984741211, 1.20000004768372],
            [73.9000015258789, 31.2000007629395, 38.9000015258789, 43.4000015258789],
        ]
    )

    result_value = 59.4272662886343

    src_idxs = numpy.array(
        [[find_nearest_index(ll.read(1), value) for value in row] for row in src_values]
    )
    src_lonlats = numpy.array(
        [[ll.xy(idx[0], idx[1]) for idx in row] for row in src_idxs]
    )
    src_xys = numpy.array(
        [[transformer.transform(lon, lat) for lon, lat in row] for row in src_lonlats]
    )
    src_xys = src_xys.reshape(-1, 2)

    result_idx = find_nearest_index(lcc.read(1), result_value)
    result_xy = lcc.xy(result_idx[0], result_idx[1])
    result_lonlat = rev_transformer.transform(result_xy[0], result_xy[1])

    nXRadius, nYRadius = 2, 2 # filter radius (4x4 kernel)

    ll_x0, ll_y0 = ll.xy(0, 0)
    lcc_xMin, lcc_yMax, lcc_xMax, lcc_yMin = get_lcc_bounds(lcc, rev_transformer)

    dfMinXOut = ((lcc_xMin - ll_x0) / 0.25) + 0.5
    dfMinYOut = ((ll_y0 - lcc_yMax) / 0.25) + 0.5
    dfMaxXOut = ((lcc_xMax - ll_x0) / 0.25) + 0.5
    dfMaxYOut = ((ll_y0 - lcc_yMin) / 0.25) + 0.5

    assert round(dfMinXOut, 5) == round(4.7102160316373727, 5)
    assert round(dfMinYOut, 5) == round(7.1311345893376767, 5)
    assert round(dfMaxXOut, 5) == round(25.909471574464362, 5)
    assert round(dfMaxYOut, 5) == round(28.519130649598168, 5)

    nMinXOutClamped = int(max(0.0, dfMinXOut))
    nMinYOutClamped = int(max(0.0, dfMinYOut))
    nMaxXOutClamped = int(min(numpy.ceil(dfMaxXOut), lcc.width))
    nMaxYOutClamped = int(min(numpy.ceil(dfMaxYOut), lcc.height))

    assert nMinXOutClamped == 4
    assert nMinYOutClamped == 7
    assert nMaxXOutClamped == 26
    assert nMaxYOutClamped == 29

    nRasterXSize, nRasterYSize = ll.width, ll.height
    
    nSrcXOff = max(0, min(nMinXOutClamped - nXRadius, nRasterXSize))
    nSrcXSize = max(0, min(nRasterXSize - nSrcXOff, nMaxXOutClamped - nSrcXOff + nXRadius))
    nSrcYOff = max(0, min(nMinYOutClamped - nYRadius, nRasterYSize))
    nSrcYSize = max(0, min(nRasterYSize - nSrcYOff, nMaxYOutClamped - nSrcYOff + nYRadius))

    assert nSrcXOff == 2
    assert nSrcXSize == 26
    assert nSrcYOff == 5
    assert nSrcYSize == 25

    padfX, padfY = padfTransform(0, 0, rev_transformer, ll_x0, ll_y0) # index of current destination pixel
    dfSrcX, dfSrcY = padfX - nSrcXOff, padfY - nSrcYOff

    iSrcX = int(numpy.floor(dfSrcX - 0.5))
    iSrcY = int(numpy.floor(dfSrcY - 0.5))

    assert iSrcX == 2
    assert iSrcY == 3

    dfSrcXSizeRaw = max(
        0.0, min(float(nRasterXSize - nMinXOutClamped), dfMaxXOut - dfMinXOut)
    )
    dfSrcYSizeRaw = max(
        0.0, min(float(nRasterYSize - nMinYOutClamped), dfMaxYOut - dfMinYOut)
    )

    assert round(dfSrcXSizeRaw, 5) == round(21.19925554282699, 5)
    assert round(dfSrcYSizeRaw, 5) == round(21.387995060260491, 5)


    dfDeltaX = dfSrcX - 0.5 - iSrcX
    dfDeltaY = dfSrcY - 0.5 - iSrcY

    assert round(dfDeltaX, 5) == round(0.4631360871996435, 5)
    assert round(dfDeltaY, 5) == round(0.54147976943991694, 5)

    dfSrcXExtraSize = nSrcXSize - dfSrcXSizeRaw
    dfSrcYExtraSize = nSrcYSize - dfSrcYSizeRaw
    
    assert round(dfSrcXExtraSize, 5) == round(4.800744457173011, 5)
    assert round(dfSrcYExtraSize, 5) == round(3.612004939739509, 5)

    xmin = 2320000
    ymin = 5090000
    xmax = 2740000
    ymax = 5640000

    nDstXSize = (xmax - xmin) / 10000
    nDstYSize = (ymax - ymin) / 10000

    assert nDstXSize == 42
    assert nDstYSize == 55

    dfXScale = float(nDstXSize) / (nSrcXSize - dfSrcXExtraSize)
    dfYScale = float(nDstYSize) / (nSrcYSize - dfSrcYExtraSize)

    assert round(dfXScale, 5) == round(1.9812016471593117, 5)
    assert round(dfYScale, 5) == round(2.5715359141192087, 5)

    # variables changing with each pixel in the destination raster
    # dfSrcX, dfSrcY, dfDeltaX, dfDeltaY
    
if __name__ == "__main__":
    main()