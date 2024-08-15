use anyhow::Result;
use float_cmp::assert_approx_eq;
use mappers::{
    projections::{LambertConformalConic, LongitudeLatitude},
    Ellipsoid,
};
use ndarray::{s, Array2, Zip};
use notgdalwarp::{CubicBSpline, RasterBounds, Warper};

mod utils;
use utils::*;

#[test]
fn waves_unchecked() -> Result<()> {
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
    let target_raster = warper.warp_unchecked(&source_raster);

    assert_eq!(target_raster.shape(), ref_raster.shape());
    Zip::from(&target_raster)
        .and(&ref_raster)
        .map_collect(|&f, &o| assert_approx_eq!(f64, f, o, epsilon = 1e-6));

    Ok(())
}

#[test]
fn nan_ignore() -> Result<()> {
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

    let mut source_raster: Array2<f64> = open_nc_data("./tests/data/waves_34.nc")?;
    source_raster.slice_mut(s![13..15, 13..15]).fill(f64::NAN);
    source_raster.slice_mut(s![22..24, 18..20]).fill(f64::NAN);
    source_raster.slice_mut(s![18..25, 19..24]).fill(f64::NAN);
    source_raster.slice_mut(s![13..15, 21..23]).fill(f64::NAN);

    let target_raster = warper.warp_ignore_nodata(&source_raster)?;
    let ref_raster: Array2<f64> = open_nc_data("./tests/data/waves_nan_ignore_ref.nc")?;

    assert_eq!(target_raster.shape(), ref_raster.shape());
    Zip::from(&target_raster)
        .and(&ref_raster)
        .map_collect(|&f, &o| assert_approx_eq!(f64, f, o, epsilon = 1e-6));

    Ok(())
}

#[test]
fn nan_reject() -> Result<()> {
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

    // should work
    let source_raster: Array2<f64> = open_nc_data("./tests/data/waves_34.nc")?;
    let ref_raster: Array2<f64> = open_nc_data("./tests/data/waves_ref.nc")?;
    let target_raster = warper.warp_reject_nodata(&source_raster)?;

    assert_eq!(target_raster.shape(), ref_raster.shape());
    Zip::from(&target_raster)
        .and(&ref_raster)
        .map_collect(|&f, &o| assert_approx_eq!(f64, f, o, epsilon = 1e-6));

    // should fail
    let mut source_raster: Array2<f64> = open_nc_data("./tests/data/waves_34.nc")?;
    source_raster.slice_mut(s![14..15, 18..19]).fill(f64::NAN);

    let target_raster = warper.warp_reject_nodata(&source_raster);
    assert!(target_raster.is_err());

    Ok(())
}

#[test]
fn nan_discard() -> Result<()> {
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

    let mut source_raster: Array2<f64> = open_nc_data("./tests/data/waves_34.nc")?;
    source_raster.slice_mut(s![13..15, 13..15]).fill(f64::NAN);
    source_raster.slice_mut(s![22..24, 18..20]).fill(f64::NAN);
    source_raster.slice_mut(s![18..25, 19..24]).fill(f64::NAN);
    source_raster.slice_mut(s![13..15, 21..23]).fill(f64::NAN);

    let target_raster = warper.warp_discard_nodata(&source_raster)?;
    let ref_raster: Array2<f64> = open_nc_data("./tests/data/waves_nan_discard_ref.nc")?;

    assert_eq!(target_raster.shape(), ref_raster.shape());
    Zip::from(&target_raster)
        .and(&ref_raster)
        .map_collect(|&f, &o| assert_approx_eq!(f64, f, o, epsilon = 1e-6));

    Ok(())
}

#[test]
fn non_finite_result() -> Result<()> {
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

    let mut source_raster: Array2<f64> = open_nc_data("./tests/data/waves_34.nc")?;
    source_raster.slice_mut(s![13..15, 21..23]).fill(f64::MAX);

    assert!(warper.warp_discard_nodata(&source_raster).is_err());
    assert!(warper.warp_reject_nodata(&source_raster).is_err());
    assert!(warper.warp_ignore_nodata(&source_raster).is_err());

    Ok(())
}
