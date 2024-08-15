use anyhow::Result;
use float_cmp::assert_approx_eq;
use mappers::{
    projections::{AzimuthalEquidistant, LambertConformalConic, LongitudeLatitude},
    Ellipsoid,
};
use ndarray::{Array2, Zip};
use notgdalwarp::{raster_constant_pad, CubicBSpline, MitchellNetravali, RasterBounds, Warper};

mod utils;
use utils::*;

#[test]
fn waves() -> Result<()> {
    let src_proj = LongitudeLatitude;
    let tgt_proj =
        LambertConformalConic::new(80., 24., 12.472955, 35.1728044444444, Ellipsoid::WGS84)?;

    let source_bounds = RasterBounds::new((60.00, 68.25), (31.75, 40.0), 0.25, 0.25, src_proj)?;
    let target_bounds = RasterBounds::new(
        (2_320_000. - 4_000_000., 2_740_000. - 4_000_000.),
        (5_090_000. - 4_000_000., 5_640_000. - 4_000_000.),
        10_000.,
        10_000.,
        tgt_proj,
    )?;

    let warper = Warper::initialize::<CubicBSpline, LongitudeLatitude, LambertConformalConic>(
        &source_bounds,
        &target_bounds,
    )?;

    let source_raster: Array2<f64> = open_nc_data("./tests/data/waves_34.nc")?;
    let ref_raster: Array2<f64> = open_nc_data("./tests/data/waves_ref.nc")?;
    let target_raster = warper.warp_ignore_nodata(&source_raster)?;

    assert_eq!(target_raster.shape(), ref_raster.shape());
    Zip::from(&target_raster)
        .and(&ref_raster)
        .map_collect(|&f, &o| assert_approx_eq!(f64, f, o, epsilon = 1e-6));

    Ok(())
}

#[test]
fn gfs_t2m() -> Result<()> {
    let src_proj = LongitudeLatitude;
    let eu_proj = LambertConformalConic::new(10.0, 52.0, 35.0, 65.0, Ellipsoid::WGS84)?;

    let source_domain = RasterBounds::new((-70.0, 85.0), (17.0, 77.0), 0.25, 0.25, src_proj)?;
    let target_domain = RasterBounds::new(
        (-4_120_000., 3_490_000.),
        (-2_750_000., 2_640_000.),
        10_000.,
        10_000.,
        eu_proj,
    )?;

    let warper = Warper::initialize::<CubicBSpline, LongitudeLatitude, LambertConformalConic>(
        &source_domain,
        &target_domain,
    )?;
    let source_raster: Array2<f64> = open_nc_data("./tests/data/gfs_t2m.nc")?;
    let target_raster = warper.warp_ignore_nodata(&source_raster)?;

    target_raster.iter().for_each(|&v| assert!(v.is_finite()));

    assert!(target_raster.max()? <= source_raster.max()?);
    assert!(target_raster.min()? >= source_raster.min()?);

    Ok(())
}

#[test]
fn mitchell() -> Result<()> {
    let src_proj = LongitudeLatitude;
    let eu_proj = LambertConformalConic::new(10.0, 52.0, 35.0, 65.0, Ellipsoid::WGS84)?;

    let source_domain = RasterBounds::new((-70.0, 85.0), (17.0, 77.0), 0.25, 0.25, src_proj)?;
    let target_domain = RasterBounds::new(
        (-4_120_000., 3_490_000.),
        (-2_750_000., 2_640_000.),
        10_000.,
        10_000.,
        eu_proj,
    )?;

    let warper = Warper::initialize::<MitchellNetravali, LongitudeLatitude, LambertConformalConic>(
        &source_domain,
        &target_domain,
    )?;
    let source_raster: Array2<f64> = open_nc_data("./tests/data/gfs_t2m.nc")?;
    let target_raster = warper.warp_ignore_nodata(&source_raster)?;

    target_raster.iter().for_each(|&v| assert!(v.is_finite()));

    dbg!(target_raster.max()?);
    dbg!(source_raster.max()?);

    assert!(target_raster.max()? <= source_raster.max()?);
    assert!(target_raster.min()? >= source_raster.min()?);

    Ok(())
}

#[test]
fn nan_padded_waves() -> Result<()> {
    let src_proj = LongitudeLatitude;
    let tgt_proj =
        LambertConformalConic::new(80., 24., 12.472955, 35.1728044444444, Ellipsoid::WGS84)?;

    let source_bounds = RasterBounds::new((59.25, 69.00), (31.00, 40.75), 0.25, 0.25, src_proj)?;
    let target_bounds = RasterBounds::new(
        (2_320_000. - 4_000_000., 2_740_000. - 4_000_000.),
        (5_090_000. - 4_000_000., 5_640_000. - 4_000_000.),
        10_000.,
        10_000.,
        tgt_proj,
    )?;

    let warper = Warper::initialize::<CubicBSpline, LongitudeLatitude, LambertConformalConic>(
        &source_bounds,
        &target_bounds,
    )?;

    let source_raster: Array2<f64> = open_nc_data("./tests/data/waves_34.nc")?;
    let source_raster = raster_constant_pad(&source_raster, 3, f64::NAN);
    let ref_raster: Array2<f64> = open_nc_data("./tests/data/waves_ref.nc")?;
    let target_raster = warper.warp_ignore_nodata(&source_raster)?;

    assert_eq!(target_raster.shape(), ref_raster.shape());
    Zip::from(&target_raster)
        .and(&ref_raster)
        .map_collect(|&f, &o| assert_approx_eq!(f64, f, o, epsilon = 1e-6));

    Ok(())
}

#[test]
fn invalid_raster_size() -> Result<()> {
    let src_proj = LongitudeLatitude;
    let tgt_proj =
        LambertConformalConic::new(80., 24., 12.472955, 35.1728044444444, Ellipsoid::WGS84)?;

    let source_bounds = RasterBounds::new((60.00, 68.25), (31.75, 40.0), 0.25, 0.25, src_proj)?;
    let target_bounds = RasterBounds::new(
        (2_320_000. - 4_000_000., 2_740_000. - 4_000_000.),
        (5_090_000. - 4_000_000., 5_640_000. - 4_000_000.),
        10_000.,
        10_000.,
        tgt_proj,
    )?;

    let warper = Warper::initialize::<CubicBSpline, LongitudeLatitude, LambertConformalConic>(
        &source_bounds,
        &target_bounds,
    )?;

    let source_raster = Array2::zeros((3, 3));
    let result = warper.warp_ignore_nodata(&source_raster);

    assert!(result.is_err());

    Ok(())
}

#[test]
fn aeqd_to_lcc() -> Result<()> {
    let src_proj = AzimuthalEquidistant::new(19.0926, 52.3469, Ellipsoid::SPHERE)?;
    let tgt_proj = LambertConformalConic::new(
        19.0926,
        52.3469,
        52.344876525433854,
        52.34892347456614,
        Ellipsoid::WGS84,
    )?;

    let source_domain = RasterBounds::new(
        (-452500., 452500.),
        (-452500., 452500.),
        1000.,
        1000.,
        src_proj,
    )?;
    let target_domain = RasterBounds::new(
        (-449500., 449500.),
        (-449500., 449500.),
        1000.,
        1000.,
        tgt_proj,
    )?;

    let warper = Warper::initialize::<CubicBSpline, AzimuthalEquidistant, LambertConformalConic>(
        &source_domain,
        &target_domain,
    )?;

    let source_raster: Array2<f64> = open_nc_data("./tests/data/aeqd_nan.nc")?;
    let source_raster = raster_constant_pad(&source_raster, 3, f64::NAN);

    let target_raster = warper.warp_ignore_nodata(&source_raster)?;
    let ref_raster: Array2<f64> = open_nc_data("./tests/data/aeqd_ref.nc")?;

    assert_eq!(target_raster.shape(), ref_raster.shape());
    Zip::from(&target_raster)
        .and(&ref_raster)
        .map_collect(|&f, &o| assert_approx_eq!(f64, f, o, epsilon = 1e-6));

    let source_values = source_raster
        .into_iter()
        .filter(|&v| !v.is_nan())
        .collect::<Vec<f64>>();

    let target_values = target_raster
        .into_iter()
        .filter(|&v| !v.is_nan())
        .collect::<Vec<f64>>();

    target_values.iter().for_each(|v| assert!(v.is_finite()));

    let source_max = *source_values.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let source_min = *source_values.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let target_max = *target_values.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let target_min = *target_values.iter().min_by(|a, b| a.total_cmp(b)).unwrap();

    assert!(target_max <= source_max);
    assert_approx_eq!(f64, source_min, target_min, epsilon = 1e-12);

    Ok(())
}
