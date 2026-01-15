use anyhow::Result;
use float_cmp::assert_approx_eq;
use mappers::{
    Ellipsoid,
    projections::{LambertConformalConic, LongitudeLatitude},
};
use mappers_warp::{CubicBSpline, RasterBoundsDefinition, Warper, WarperError};
use ndarray::{Array2, Zip, s};

mod utils;
use utils::*;

#[test]
fn result_ok() -> Result<()> {
    let src_proj = LongitudeLatitude;
    let tgt_proj = LambertConformalConic::builder()
        .ref_lonlat(80., 24.)
        .standard_parallels(12.472955, 35.1728044444444)
        .ellipsoid(Ellipsoid::WGS84)
        .initialize_projection()?;

    let source_bounds =
        RasterBoundsDefinition::new((60.00, 68.25), (31.75, 40.0), 0.25, 0.25, src_proj)?;
    let target_bounds = RasterBoundsDefinition::new(
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

    let target_raster = warper.warp_reject_nodata(&source_raster)?;
    assert_eq!(target_raster.shape(), ref_raster.shape());
    Zip::from(&target_raster)
        .and(&ref_raster)
        .map_collect(|&f, &o| assert_approx_eq!(f64, f, o, epsilon = 1e-6));

    let target_raster = warper.warp_ignore_nodata(&source_raster)?;
    assert_eq!(target_raster.shape(), ref_raster.shape());
    Zip::from(&target_raster)
        .and(&ref_raster)
        .map_collect(|&f, &o| assert_approx_eq!(f64, f, o, epsilon = 1e-6));

    let target_raster = warper.warp_discard_nodata(&source_raster)?;
    assert_eq!(target_raster.shape(), ref_raster.shape());
    Zip::from(&target_raster)
        .and(&ref_raster)
        .map_collect(|&f, &o| assert_approx_eq!(f64, f, o, epsilon = 1e-6));

    #[cfg(feature = "multithreading")]
    {
        let target_raster = warper.warp_unchecked_parallel(&source_raster);
        assert_eq!(target_raster.shape(), ref_raster.shape());
        Zip::from(&target_raster)
            .and(&ref_raster)
            .map_collect(|&f, &o| assert_approx_eq!(f64, f, o, epsilon = 1e-6));

        let target_raster = warper.warp_reject_nodata_parallel(&source_raster)?;
        assert_eq!(target_raster.shape(), ref_raster.shape());
        Zip::from(&target_raster)
            .and(&ref_raster)
            .map_collect(|&f, &o| assert_approx_eq!(f64, f, o, epsilon = 1e-6));

        let target_raster = warper.warp_ignore_nodata_parallel(&source_raster)?;
        assert_eq!(target_raster.shape(), ref_raster.shape());
        Zip::from(&target_raster)
            .and(&ref_raster)
            .map_collect(|&f, &o| assert_approx_eq!(f64, f, o, epsilon = 1e-6));

        let target_raster = warper.warp_discard_nodata_parallel(&source_raster)?;
        assert_eq!(target_raster.shape(), ref_raster.shape());
        Zip::from(&target_raster)
            .and(&ref_raster)
            .map_collect(|&f, &o| assert_approx_eq!(f64, f, o, epsilon = 1e-6));
    }

    Ok(())
}

#[test]
fn single_nan_input_area() -> Result<()> {
    let src_proj = LongitudeLatitude;
    let tgt_proj = LambertConformalConic::builder()
        .ref_lonlat(80., 24.)
        .standard_parallels(12.472955, 35.1728044444444)
        .ellipsoid(Ellipsoid::WGS84)
        .initialize_projection()?;

    let source_bounds =
        RasterBoundsDefinition::new((60.00, 68.25), (31.75, 40.0), 0.25, 0.25, src_proj)?;
    let target_bounds = RasterBoundsDefinition::new(
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
    // source_raster.slice_mut(s![14..15, 18..19]).fill(f64::NAN);
    source_raster[[0,0]] = f64::NAN;

    dbg!(&source_raster);

    let _ = warper.warp_unchecked(&source_raster);

    let target_raster = warper.warp_reject_nodata(&source_raster).unwrap_err();
    assert!(matches!(target_raster, WarperError::NanError));

    let target_raster = warper.warp_ignore_nodata(&source_raster);
    assert!(target_raster.is_ok());

    let target_raster = warper.warp_discard_nodata(&source_raster);
    assert!(target_raster.is_ok());

    #[cfg(feature = "multithreading")]
    {
        let _ = warper.warp_unchecked_parallel(&source_raster);

        let target_raster = warper
            .warp_reject_nodata_parallel(&source_raster)
            .unwrap_err();
        assert!(matches!(target_raster, WarperError::NanError));

        let target_raster = warper.warp_ignore_nodata_parallel(&source_raster);
        assert!(target_raster.is_ok());

        let target_raster = warper.warp_discard_nodata_parallel(&source_raster);
        assert!(target_raster.is_ok());
    }

    Ok(())
}

// #[test]
// fn multi_nan_input_area() -> Result<()> {
//     let src_proj = LongitudeLatitude;
//     let tgt_proj = LambertConformalConic::builder()
//         .ref_lonlat(80., 24.)
//         .standard_parallels(12.472955, 35.1728044444444)
//         .ellipsoid(Ellipsoid::WGS84)
//         .initialize_projection()?;

//     let source_bounds =
//         RasterBoundsDefinition::new((60.00, 68.25), (31.75, 40.0), 0.25, 0.25, src_proj)?;
//     let target_bounds = RasterBoundsDefinition::new(
//         (2_320_000. - 4_000_000., 2_740_000. - 4_000_000.),
//         (5_090_000. - 4_000_000., 5_640_000. - 4_000_000.),
//         10_000.,
//         10_000.,
//         tgt_proj,
//     )?;

//     let warper = Warper::initialize::<CubicBSpline, LongitudeLatitude, LambertConformalConic>(
//         &source_bounds,
//         &target_bounds,
//     )?;

//     let mut source_raster: Array2<f64> = open_nc_data("./tests/data/waves_34.nc")?;
//     source_raster.slice_mut(s![13..15, 13..15]).fill(f64::NAN);
//     source_raster.slice_mut(s![22..24, 18..20]).fill(f64::NAN);
//     source_raster.slice_mut(s![18..25, 19..24]).fill(f64::NAN);
//     source_raster.slice_mut(s![13..15, 21..23]).fill(f64::NAN);

//     let _ = warper.warp_unchecked(&source_raster);

//     let target_raster = warper.warp_reject_nodata(&source_raster).unwrap_err();
//     assert!(matches!(target_raster, WarperError::NanError));

//     let target_raster = warper.warp_ignore_nodata(&source_raster);
//     assert!(target_raster.is_ok());

//     let target_raster = warper.warp_discard_nodata(&source_raster);
//     assert!(target_raster.is_ok());

//     #[cfg(feature = "multithreading")]
//     {
//         let _ = warper.warp_unchecked_parallel(&source_raster);

//         let target_raster = warper
//             .warp_reject_nodata_parallel(&source_raster)
//             .unwrap_err();
//         assert!(matches!(target_raster, WarperError::NanError));

//         let target_raster = warper.warp_ignore_nodata_parallel(&source_raster);
//         assert!(target_raster.is_ok());

//         let target_raster = warper.warp_discard_nodata_parallel(&source_raster);
//         assert!(target_raster.is_ok());
//     }

//     Ok(())
// }

#[test]
fn non_finite_result() -> Result<()> {
    let src_proj = LongitudeLatitude;
    let tgt_proj = LambertConformalConic::builder()
        .ref_lonlat(80., 24.)
        .standard_parallels(12.472955, 35.1728044444444)
        .ellipsoid(Ellipsoid::WGS84)
        .initialize_projection()?;

    let source_bounds =
        RasterBoundsDefinition::new((60.00, 68.25), (31.75, 40.0), 0.25, 0.25, src_proj)?;
    let target_bounds = RasterBoundsDefinition::new(
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
    source_raster
        .slice_mut(s![13..15, 21..23])
        .fill(f64::INFINITY);

    let _ = warper.warp_unchecked(&source_raster);

    assert!(matches!(
        warper.warp_discard_nodata(&source_raster).unwrap_err(),
        WarperError::WarpingError
    ));
    assert!(matches!(
        warper.warp_reject_nodata(&source_raster).unwrap_err(),
        WarperError::WarpingError
    ));
    assert!(matches!(
        warper.warp_ignore_nodata(&source_raster).unwrap_err(),
        WarperError::WarpingError
    ));

    #[cfg(feature = "multithreading")]
    {
        let _ = warper.warp_unchecked_parallel(&source_raster);

        assert!(matches!(
            warper
                .warp_discard_nodata_parallel(&source_raster)
                .unwrap_err(),
            WarperError::WarpingError
        ));
        assert!(matches!(
            warper
                .warp_reject_nodata_parallel(&source_raster)
                .unwrap_err(),
            WarperError::WarpingError
        ));
        assert!(matches!(
            warper
                .warp_ignore_nodata_parallel(&source_raster)
                .unwrap_err(),
            WarperError::WarpingError
        ));
    }

    Ok(())
}

#[test]
fn init_error() -> Result<()> {
    let src_proj = LongitudeLatitude;
    let tgt_proj = LambertConformalConic::builder()
        .ref_lonlat(80., 24.)
        .standard_parallels(12.472955, 35.1728044444444)
        .ellipsoid(Ellipsoid::WGS84)
        .initialize_projection()?;

    let source_bounds =
        RasterBoundsDefinition::new((61.00, 67.00), (31.75, 40.0), 0.25, 0.25, src_proj)?;
    let target_bounds = RasterBoundsDefinition::new(
        (2_320_000. - 4_000_000., 2_740_000. - 4_000_000.),
        (5_090_000. - 4_000_000., 5_640_000. - 4_000_000.),
        10_000.,
        10_000.,
        tgt_proj,
    )?;

    let result = Warper::initialize::<CubicBSpline, LongitudeLatitude, LambertConformalConic>(
        &source_bounds,
        &target_bounds,
    )
    .unwrap_err();

    assert!(matches!(result, WarperError::SourceRasterTooSmall));

    Ok(())
}

#[test]
fn invalid_input_shape() -> Result<()> {
    let src_proj = LongitudeLatitude;
    let tgt_proj = LambertConformalConic::builder()
        .ref_lonlat(80., 24.)
        .standard_parallels(12.472955, 35.1728044444444)
        .ellipsoid(Ellipsoid::WGS84)
        .initialize_projection()?;

    let source_bounds =
        RasterBoundsDefinition::new((60.00, 68.25), (31.75, 40.0), 0.25, 0.25, src_proj)?;
    let target_bounds = RasterBoundsDefinition::new(
        (2_320_000. - 4_000_000., 2_740_000. - 4_000_000.),
        (5_090_000. - 4_000_000., 5_640_000. - 4_000_000.),
        10_000.,
        10_000.,
        tgt_proj,
    )?;

    let warper = Warper::initialize::<CubicBSpline, _, _>(&source_bounds, &target_bounds)?;

    // valid shape: (34, 34)
    let invalid_source_rasters = vec![
        Array2::zeros((10, 10)),
        Array2::zeros((34, 10)),
        Array2::zeros((10, 34)),
    ];

    for source_raster in invalid_source_rasters {
        assert!(matches!(
            warper.warp_reject_nodata(&source_raster).unwrap_err(),
            WarperError::InvalidRasterDimensions
        ));
        assert!(matches!(
            warper.warp_ignore_nodata(&source_raster).unwrap_err(),
            WarperError::InvalidRasterDimensions
        ));
        assert!(matches!(
            warper.warp_discard_nodata(&source_raster).unwrap_err(),
            WarperError::InvalidRasterDimensions
        ));

        #[cfg(feature = "multithreading")]
        assert!(matches!(
            warper
                .warp_reject_nodata_parallel(&source_raster)
                .unwrap_err(),
            WarperError::InvalidRasterDimensions
        ));

        #[cfg(feature = "multithreading")]
        assert!(matches!(
            warper
                .warp_ignore_nodata_parallel(&source_raster)
                .unwrap_err(),
            WarperError::InvalidRasterDimensions
        ));

        #[cfg(feature = "multithreading")]
        assert!(matches!(
            warper
                .warp_discard_nodata_parallel(&source_raster)
                .unwrap_err(),
            WarperError::InvalidRasterDimensions
        ));
    }

    Ok(())
}

#[test]
#[should_panic = "Slice end 11 is past end of axis of length 10"]
fn invalid_input_shape_unchecked() {
    let src_proj = LongitudeLatitude;
    let tgt_proj = LambertConformalConic::builder()
        .ref_lonlat(80., 24.)
        .standard_parallels(12.472955, 35.1728044444444)
        .ellipsoid(Ellipsoid::WGS84)
        .initialize_projection()
        .unwrap();

    let source_bounds =
        RasterBoundsDefinition::new((60.00, 68.25), (31.75, 40.0), 0.25, 0.25, src_proj).unwrap();
    let target_bounds = RasterBoundsDefinition::new(
        (2_320_000. - 4_000_000., 2_740_000. - 4_000_000.),
        (5_090_000. - 4_000_000., 5_640_000. - 4_000_000.),
        10_000.,
        10_000.,
        tgt_proj,
    )
    .unwrap();

    let warper = Warper::initialize::<CubicBSpline, _, _>(&source_bounds, &target_bounds).unwrap();

    // valid shape: (34, 34)
    let invalid_source_raster = Array2::<f64>::zeros((10, 10));
    let _ = warper.warp_unchecked(&invalid_source_raster);
}

#[test]
#[should_panic = "Slice end 11 is past end of axis of length 10"]
#[cfg(feature = "multithreading")]
fn invalid_input_shape_unchecked_parallel() {
    let src_proj = LongitudeLatitude;
    let tgt_proj = LambertConformalConic::builder()
        .ref_lonlat(80., 24.)
        .standard_parallels(12.472955, 35.1728044444444)
        .ellipsoid(Ellipsoid::WGS84)
        .initialize_projection()
        .unwrap();

    let source_bounds =
        RasterBoundsDefinition::new((60.00, 68.25), (31.75, 40.0), 0.25, 0.25, src_proj).unwrap();
    let target_bounds = RasterBoundsDefinition::new(
        (2_320_000. - 4_000_000., 2_740_000. - 4_000_000.),
        (5_090_000. - 4_000_000., 5_640_000. - 4_000_000.),
        10_000.,
        10_000.,
        tgt_proj,
    )
    .unwrap();

    let warper = Warper::initialize::<CubicBSpline, _, _>(&source_bounds, &target_bounds).unwrap();

    // valid shape: (34, 34)
    let invalid_source_raster = Array2::<f64>::zeros((10, 10));
    let _ = warper.warp_unchecked_parallel(&invalid_source_raster);
}
