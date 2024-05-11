use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mappers::{
    projections::{LambertConformalConic, LongitudeLatitude},
    Ellipsoid,
};
use ndarray::Array2;
use notgdalwarp::{CubicBSpline, RasterBounds, Warper};

pub fn criterion_benchmark(c: &mut Criterion) {
    let src_proj = LongitudeLatitude;
    let eu_proj = LambertConformalConic::new(10.0, 52.0, 35.0, 65.0, Ellipsoid::WGS84).unwrap();

    let source_domain =
        RasterBounds::new((-70.0, 85.0), (17.0, 77.0), 0.25, 0.25, src_proj).unwrap();
    let target_domain = RasterBounds::new(
        (-4_120_000., 3_490_000.),
        (-2_750_000., 2_640_000.),
        10_000.,
        10_000.,
        eu_proj,
    )
    .unwrap();

    let warper = Warper::initialize::<CubicBSpline, LongitudeLatitude, LambertConformalConic>(
        &source_domain,
        &target_domain,
    )
    .unwrap();
    let source_raster: Array2<f64> = ndarray_npy::read_npy("./tests/data/gfs_t2m.npy").unwrap();

    c.bench_function("warp only", |b| {
        b.iter(|| warper.warp(black_box(&source_raster)))
    });

    // error check
    let _ = warper.warp(&source_raster).unwrap();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
