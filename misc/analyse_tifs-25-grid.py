import numpy
import pyproj
import rasterio


def padfTransform(lcc_i, lcc_j, rev_transformer, ll_x0, ll_y0):
    # corners of EDGES of target raster
    xmin = 2320000 - 5000
    ymax = 5640000 + 5000

    dx = 10000
    dy = 10000

    # shift here is to get xy value of the center of the pixel
    x, y = lcc_i + 0.5, lcc_j + 0.5
    lcc_x = xmin + x * dx
    lcc_y = ymax - y * dy
    lcc_lonlat = rev_transformer.transform(lcc_x, lcc_y)

    # shifts are a result of using lonlat vaues in ll as midpoints
    ll_x0 = ll_x0 - (0.25 / 2)
    ll_y0 = ll_y0 + (0.25 / 2)

    # scaling is 1/dx, 1/dy
    padfX, padfY = (lcc_lonlat[0] - ll_x0) * 4, (ll_y0 - lcc_lonlat[1]) * 4

    return padfX, padfY


def find_nearest_index(array, value):
    array = numpy.asarray(array)
    idx = (numpy.abs(array - value)).argmin()

    idx = numpy.unravel_index(idx, array.shape)

    return idx


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


def get_lcc_bounds(rev_transformer):
    # corners gridpoints of target
    xmin = 2320000
    ymin = 5090000
    xmax = 2740000
    ymax = 5640000

    dx = 10000
    dy = 10000

    xmin, ymin = xmin - dx / 2, ymin - dy / 2
    xmax, ymax = xmax + dx / 2, ymax + dy / 2

    upper_edge_x = numpy.arange(xmin, xmax, dx)
    upper_edge_y = numpy.full_like(upper_edge_x, ymax)
    upper_edge_xy = numpy.array(list(zip(upper_edge_x, upper_edge_y)))

    right_edge_y = numpy.arange(ymax, ymin, -dy)
    right_edge_x = numpy.full_like(right_edge_y, xmax)
    right_edge_xy = numpy.array(list(zip(right_edge_x, right_edge_y)))

    lower_edge_x = numpy.arange(xmax, xmin, -dx)
    lower_edge_y = numpy.full_like(lower_edge_x, ymin)
    lower_edge_xy = numpy.array(list(zip(lower_edge_x, lower_edge_y)))

    left_edge_y = numpy.arange(ymin, ymax, dy)
    left_edge_x = numpy.full_like(left_edge_y, xmin)
    left_edge_xy = numpy.array(list(zip(left_edge_x, left_edge_y)))

    all_cords = numpy.concatenate(
        (upper_edge_xy, right_edge_xy, lower_edge_xy, left_edge_xy)
    )

    lonlats = numpy.array([rev_transformer.transform(x, y) for x, y in all_cords])

    xMin = numpy.min(lonlats[:, 0])
    xMax = numpy.max(lonlats[:, 0])
    yMin = numpy.min(lonlats[:, 1])
    yMax = numpy.max(lonlats[:, 1])

    return xMin, yMax, xMax, yMin


def main():
    ll = rasterio.open("./misc/lonlat-p25.tif")
    lcc = rasterio.open("./misc/lcc-india-p25-grid.tif")

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
    _result_lonlat = rev_transformer.transform(result_xy[0], result_xy[1])

    nXRadius, nYRadius = 2, 2  # filter radius (4x4 kernel)

    ll_x0, ll_y0 = ll.xy(0, 0)
    lcc_xMin, lcc_yMax, lcc_xMax, lcc_yMin = get_lcc_bounds(rev_transformer)

    dfMinXOut = ((lcc_xMin - ll_x0) / 0.25) + 0.5
    dfMinYOut = ((ll_y0 - lcc_yMax) / 0.25) + 0.5
    dfMaxXOut = ((lcc_xMax - ll_x0) / 0.25) + 0.5
    dfMaxYOut = ((ll_y0 - lcc_yMin) / 0.25) + 0.5

    assert round(dfMinXOut, 5) == round(4.457122955747991, 5)
    assert round(dfMinYOut, 5) == round(6.9363298550977959, 5)
    assert round(dfMaxXOut, 5) == round(26.145584743939651, 5)
    assert round(dfMaxYOut, 5) == round(28.72260733112293, 5)

    nMinXOutClamped = int(max(0.0, dfMinXOut))
    nMinYOutClamped = int(max(0.0, dfMinYOut))
    nMaxXOutClamped = int(min(numpy.ceil(dfMaxXOut), lcc.width))
    nMaxYOutClamped = int(min(numpy.ceil(dfMaxYOut), lcc.height))

    assert nMinXOutClamped == 4
    assert nMinYOutClamped == 6
    assert nMaxXOutClamped == 27
    assert nMaxYOutClamped == 29

    nRasterXSize, nRasterYSize = ll.width, ll.height

    nSrcXOff = max(0, min(nMinXOutClamped - nXRadius, nRasterXSize))
    nSrcXSize = max(
        0, min(nRasterXSize - nSrcXOff, nMaxXOutClamped - nSrcXOff + nXRadius)
    )
    nSrcYOff = max(0, min(nMinYOutClamped - nYRadius, nRasterYSize))
    nSrcYSize = max(
        0, min(nRasterYSize - nSrcYOff, nMaxYOutClamped - nSrcYOff + nYRadius)
    )

    assert nSrcXOff == 2
    assert nSrcXSize == 27
    assert nSrcYOff == 4
    assert nSrcYSize == 26

    dfSrcXSizeRaw = max(
        0.0, min(float(nRasterXSize - nMinXOutClamped), dfMaxXOut - dfMinXOut)
    )
    dfSrcYSizeRaw = max(
        0.0, min(float(nRasterYSize - nMinYOutClamped), dfMaxYOut - dfMinYOut)
    )

    assert round(dfSrcXSizeRaw, 5) == round(21.68846178819166, 5)
    assert round(dfSrcYSizeRaw, 5) == round(21.786277476025134, 5)

    dfSrcXExtraSize = nSrcXSize - dfSrcXSizeRaw
    dfSrcYExtraSize = nSrcYSize - dfSrcYSizeRaw

    assert round(dfSrcXExtraSize, 5) == round(5.3115382118083403, 5)
    assert round(dfSrcYExtraSize, 5) == round(4.2137225239748659, 5)

    padfX, padfY = padfTransform(
        0, 0, rev_transformer, ll_x0, ll_y0
    )  # src coords of current target pixel

    assert round(padfX, 5) == round(4.7102160316373727, 5)
    assert round(padfY, 5) == round(8.8887293250701873, 5)

    dfSrcX, dfSrcY = padfX - nSrcXOff, padfY - nSrcYOff

    assert round(dfSrcX, 5) == round(2.7102160316373727, 5)
    assert round(dfSrcY, 5) == round(4.8887293250701873, 5)

    iSrcX = int(numpy.floor(dfSrcX - 0.5))
    iSrcY = int(numpy.floor(dfSrcY - 0.5))

    assert iSrcX == 2
    assert iSrcY == 4

    dfDeltaX = dfSrcX - 0.5 - iSrcX
    dfDeltaY = dfSrcY - 0.5 - iSrcY

    assert round(dfDeltaX, 5) == round(0.21021603163737268, 5)
    assert round(dfDeltaY, 5) == round(0.38872932507018731, 5)

    xmin = 2320000
    ymin = 5090000
    xmax = 2740000
    ymax = 5640000

    nDstXSize = (xmax - xmin) / 10000 + 1
    nDstYSize = (ymax - ymin) / 10000 + 1

    assert nDstXSize == 43
    assert nDstYSize == 56

    dfXScale = float(nDstXSize) / (nSrcXSize - dfSrcXExtraSize)
    dfYScale = float(nDstYSize) / (nSrcYSize - dfSrcYExtraSize)

    assert round(dfXScale, 5) == round(1.9826210092691527, 5)
    assert round(dfYScale, 5) == round(2.5704253542912783, 5)

    # variables changing with each pixel in the destination raster
    # dfSrcX, dfSrcY, dfDeltaX, dfDeltaY


if __name__ == "__main__":
    main()


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
