use float_cmp::assert_approx_eq;
use mappers::{projections::LambertConformalConic, Ellipsoid};
use ndarray::{Array2, Zip};
use not_gdalwarp::{CubicBSpline, RasterBounds, Warper};

#[test]
fn waves() {
    let source_bounds = RasterBounds::new((60.00, 68.25), (31.75, 40.0), 0.25, 0.25).unwrap();

    let target_bounds = RasterBounds::new(
        (2_320_000. - 4_000_000., 2_740_000. - 4_000_000.),
        (5_090_000. - 4_000_000., 5_640_000. - 4_000_000.),
        10_000.,
        10_000.,
    )
    .unwrap();

    let proj = LambertConformalConic::new(80., 24., 12.472955, 35.1728044444444, Ellipsoid::WGS84)
        .unwrap();

    let warper = Warper::initialize::<LambertConformalConic, CubicBSpline>(
        &source_bounds,
        &target_bounds,
        &proj,
    )
    .unwrap();

    let source_raster: Array2<f64> = ndarray_npy::read_npy("./test-data/waves_34.npy").unwrap();
    let ref_raster: Array2<f64> = ndarray_npy::read_npy("./test-data/waves_ref.npy").unwrap();
    let target_raster = warper.warp(&source_raster).unwrap();

    assert_eq!(target_raster.shape(), ref_raster.shape());
    Zip::from(&target_raster)
        .and(&ref_raster)
        .map_collect(|&f, &o| assert_approx_eq!(f64, f, o, epsilon = 1e-6));
}
