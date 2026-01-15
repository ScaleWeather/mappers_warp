use core::f64;

use anyhow::Result;
use mappers::{
    Ellipsoid,
    projections::{LambertConformalConic, LongitudeLatitude},
};
use mappers_warp::{CubicBSpline, RasterBoundsDefinition, Warper, WarperError};
use ndarray::{Array2, s};

mod utils;
use utils::*;

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

    assert!(warper.warp_discard_nodata(&source_raster).is_err());
    assert!(warper.warp_reject_nodata(&source_raster).is_err());
    assert!(warper.warp_ignore_nodata(&source_raster).is_err());

    #[cfg(feature = "multithreading")]
    {
        assert!(warper.warp_discard_nodata_parallel(&source_raster).is_err());
        assert!(warper.warp_reject_nodata_parallel(&source_raster).is_err());
        assert!(warper.warp_ignore_nodata_parallel(&source_raster).is_err());
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
