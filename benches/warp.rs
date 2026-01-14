use std::{hint::black_box, time::Duration};

use anyhow::{Context, Result};
use criterion::{Criterion, criterion_group, criterion_main};
use mappers::{
    Ellipsoid,
    projections::{LambertConformalConic, LongitudeLatitude},
};
use mappers_warp::{CubicBSpline, RasterBoundsDefinition, Warper};

pub fn criterion_benchmark(c: &mut Criterion) {
    inner_bench(c).unwrap()
}

pub fn inner_bench(c: &mut Criterion) -> Result<()> {
    let src_proj = LongitudeLatitude;
    let eu_proj = LambertConformalConic::builder()
        .ref_lonlat(10., 52.)
        .standard_parallels(35., 65.)
        .ellipsoid(Ellipsoid::WGS84)
        .initialize_projection()?;

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

    // error check
    let _ = warper.warp_ignore_nodata(&source_raster)?;
    let _ = warper.warp_discard_nodata(&source_raster)?;
    let _ = warper.warp_reject_nodata(&source_raster)?;

    let mut init_group = c.benchmark_group("Initializer");
    init_group.warm_up_time(Duration::from_secs(5));
    init_group.measurement_time(Duration::from_secs(10));

    init_group.bench_function("Serial", |b| {
        b.iter(|| {
            Warper::initialize::<CubicBSpline, _, _>(
                black_box(&source_domain),
                black_box(&target_domain),
            )
        })
    });
    #[cfg(feature = "multithreading")]
    init_group.bench_function("Parallel", |b| {
        b.iter(|| {
            Warper::initialize_parallel::<CubicBSpline, _, _>(
                black_box(&source_domain),
                black_box(&target_domain),
            )
        })
    });
    init_group.finish();

    let mut unchecked_group = c.benchmark_group("Warp Unchecked");
    unchecked_group.warm_up_time(Duration::from_secs(5));
    unchecked_group.measurement_time(Duration::from_secs(10));

    unchecked_group.bench_function("Serial", |b| {
        b.iter(|| warper.warp_unchecked(black_box(&source_raster)))
    });
    #[cfg(feature = "multithreading")]
    unchecked_group.bench_function("Parallel", |b| {
        b.iter(|| warper.warp_unchecked_parallel(black_box(&source_raster)))
    });
    unchecked_group.finish();

    let mut reject_group = c.benchmark_group("Warp Reject");
    reject_group.warm_up_time(Duration::from_secs(5));
    reject_group.measurement_time(Duration::from_secs(10));

    reject_group.bench_function("Serial", |b| {
        b.iter(|| warper.warp_reject_nodata(black_box(&source_raster)))
    });
    #[cfg(feature = "multithreading")]
    reject_group.bench_function("Parallel", |b| {
        b.iter(|| warper.warp_reject_nodata_parallel(black_box(&source_raster)))
    });
    reject_group.finish();

    let mut discard_group = c.benchmark_group("Warp Discard");
    discard_group.warm_up_time(Duration::from_secs(5));
    discard_group.measurement_time(Duration::from_secs(10));

    discard_group.bench_function("Serial", |b| {
        b.iter(|| warper.warp_discard_nodata(black_box(&source_raster)))
    });
    #[cfg(feature = "multithreading")]
    discard_group.bench_function("Parallel", |b| {
        b.iter(|| warper.warp_discard_nodata_parallel(black_box(&source_raster)))
    });
    discard_group.finish();

    let mut ignore_group = c.benchmark_group("Warp Ignore");
    ignore_group.warm_up_time(Duration::from_secs(5));
    ignore_group.measurement_time(Duration::from_secs(10));

    ignore_group.bench_function("Serial", |b| {
        b.iter(|| warper.warp_ignore_nodata(black_box(&source_raster)))
    });
    #[cfg(feature = "multithreading")]
    ignore_group.bench_function("Parallel", |b| {
        b.iter(|| warper.warp_ignore_nodata_parallel(black_box(&source_raster)))
    });
    ignore_group.finish();

    Ok(())
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
