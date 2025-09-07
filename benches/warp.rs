use anyhow::{Context, Result};
use criterion::{Criterion, criterion_group, criterion_main};
use mappers::{
    Ellipsoid,
    projections::{LambertConformalConic, LongitudeLatitude},
};
use mappers_warp::{CubicBSpline, RasterBoundsDefinition, Warper};
use std::hint::black_box;

pub fn criterion_benchmark(c: &mut Criterion) {
    inner_bench(c).unwrap()
}

pub fn inner_bench(c: &mut Criterion) -> Result<()> {
    let src_proj = LongitudeLatitude;
    let eu_proj = LambertConformalConic::new(10.0, 52.0, 35.0, 65.0, Ellipsoid::WGS84)?;

    let source_domain =
        RasterBoundsDefinition::new((-70.0, 85.0), (17.0, 77.0), 0.25, 0.25, src_proj)?;
    let target_domain = RasterBoundsDefinition::new(
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
    let source_raster = {
        let file = netcdf::open("./tests/data/gfs_t2m.nc")?;
        let var = file.variable("data_arr").context("")?;

        let data = var.get::<f64, _>(..)?;
        let data = data.into_dimensionality()?;

        data
    };

    c.bench_function("warp_unchecked", |b| {
        b.iter(|| warper.warp_unchecked(black_box(&source_raster)))
    });

    c.bench_function("warp_ignore_nodata", |b| {
        b.iter(|| warper.warp_ignore_nodata(black_box(&source_raster)))
    });

    c.bench_function("warp_discard_nodata", |b| {
        b.iter(|| warper.warp_discard_nodata(black_box(&source_raster)))
    });

    c.bench_function("warp_reject_nodata", |b| {
        b.iter(|| warper.warp_reject_nodata(black_box(&source_raster)))
    });

    // error check
    let _ = warper.warp_ignore_nodata(&source_raster)?;
    let _ = warper.warp_discard_nodata(&source_raster)?;
    let _ = warper.warp_reject_nodata(&source_raster)?;

    Ok(())
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
