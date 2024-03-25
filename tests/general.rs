use mappers::{projections::LambertConformalConic, Ellipsoid};
use ndarray::Array2;
use not_gdalwarp::{CubicBSpline, RasterBounds, Warper};

#[test]
fn random_square_33() {
    let source_bounds = RasterBounds::new((60.00, 70.0), (32.00, 40.0), 0.25, 0.25).unwrap();

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

    let source_raster: Array2<f64> =
        ndarray_npy::read_npy("./test-data/random_square_33.npy").unwrap();

    let target_raster = warper.warp(&source_raster).unwrap();

    println!("{:?}", target_raster);
}
