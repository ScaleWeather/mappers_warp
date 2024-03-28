use anyhow::Result;
use float_cmp::assert_approx_eq;
use mappers::{projections::LambertConformalConic, Ellipsoid};
use ndarray::{Array2, Zip};
use ndarray_stats::QuantileExt;
use not_gdalwarp::{CubicBSpline, MitchellNetravali, RasterBounds, Warper};

#[test]
fn waves() -> Result<()> {
    let source_bounds = RasterBounds::new((60.00, 68.25), (31.75, 40.0), 0.25, 0.25)?;

    let target_bounds = RasterBounds::new(
        (2_320_000. - 4_000_000., 2_740_000. - 4_000_000.),
        (5_090_000. - 4_000_000., 5_640_000. - 4_000_000.),
        10_000.,
        10_000.,
    )?;

    let proj = LambertConformalConic::new(80., 24., 12.472955, 35.1728044444444, Ellipsoid::WGS84)?;

    let warper = Warper::initialize::<CubicBSpline>(&source_bounds, &target_bounds, &proj)?;

    let source_raster: Array2<f64> = ndarray_npy::read_npy("./tests/data/waves_34.npy")?;
    let ref_raster: Array2<f64> = ndarray_npy::read_npy("./tests/data/waves_ref.npy")?;
    let target_raster = warper.warp(&source_raster)?;

    assert_eq!(target_raster.shape(), ref_raster.shape());
    Zip::from(&target_raster)
        .and(&ref_raster)
        .map_collect(|&f, &o| assert_approx_eq!(f64, f, o, epsilon = 1e-6));

    Ok(())
}

#[test]
fn gfs_t2m() -> Result<()> {
    let eu_proj = LambertConformalConic::new(10.0, 52.0, 35.0, 65.0, Ellipsoid::WGS84)?;
    let source_domain = RasterBounds::new((-70.0, 85.0), (17.0, 77.0), 0.25, 0.25)?;
    let target_domain = RasterBounds::new(
        (-4_120_000., 3_490_000.),
        (-2_750_000., 2_640_000.),
        10_000.,
        10_000.,
    )?;

    let warper = Warper::initialize::<CubicBSpline>(&source_domain, &target_domain, &eu_proj)?;
    let source_raster: Array2<f64> = ndarray_npy::read_npy("./tests/data/gfs_t2m.npy")?;
    let target_raster = warper.warp(&source_raster)?;

    target_raster.iter().for_each(|&v| assert!(v.is_finite()));

    assert!(target_raster.max()? <= source_raster.max()?);
    assert!(target_raster.min()? >= source_raster.min()?);

    Ok(())
}

#[test]
fn mitchell() -> Result<()> {
    let eu_proj = LambertConformalConic::new(10.0, 52.0, 35.0, 65.0, Ellipsoid::WGS84)?;
    let source_domain = RasterBounds::new((-70.0, 85.0), (17.0, 77.0), 0.25, 0.25)?;
    let target_domain = RasterBounds::new(
        (-4_120_000., 3_490_000.),
        (-2_750_000., 2_640_000.),
        10_000.,
        10_000.,
    )?;

    let warper = Warper::initialize::<MitchellNetravali>(&source_domain, &target_domain, &eu_proj)?;
    let source_raster: Array2<f64> = ndarray_npy::read_npy("./tests/data/gfs_t2m.npy")?;
    let target_raster = warper.warp(&source_raster)?;

    target_raster.iter().for_each(|&v| assert!(v.is_finite()));

    assert!(target_raster.max()? <= source_raster.max()?);
    assert!(target_raster.min()? >= source_raster.min()?);

    Ok(())
}
