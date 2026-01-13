#![cfg(feature = "multithread")]

use anyhow::Result;
use mappers::{
    Ellipsoid,
    projections::{LambertConformalConic, LongitudeLatitude},
};
use mappers_warp::{CubicBSpline, RasterBoundsDefinition, Warper};

#[test]
fn parallel_init_equivalence() -> Result<()> {
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

    let serial_warper = Warper::initialize::<CubicBSpline, _, _>(&source_bounds, &target_bounds)?;
    let prllel_warper =
        Warper::initialize_parallel::<CubicBSpline, _, _>(&source_bounds, &target_bounds)?;
    assert_eq!(serial_warper, prllel_warper);

    Ok(())
}
