# Mappers Warp

[![Github Repository](https://img.shields.io/badge/Github-Repository-blue?style=flat-square&logo=github&color=blue)](https://github.com/ScaleWeather/mappers_warp)
[![Crates.io](https://img.shields.io/crates/v/mappers_warp?style=flat-square)](https://crates.io/crates/mappers_warp)
[![License](https://img.shields.io/github/license/ScaleWeather/mappers_warp?style=flat-square)](https://choosealicense.com/licenses/apache-2.0/)
[![dependency status](https://deps.rs/repo/github/ScaleWeather/mappers_warp/status.svg?style=flat-square)](https://deps.rs/repo/github/ScaleWeather/mappers_warp)
[![docs.rs](https://img.shields.io/docsrs/mappers_warp?style=flat-square)](https://docs.rs/mappers_warp)

Very simplistic tool for reprojecting maps, based on the `GdalWarp`, using mappers for geographic projection.

This tool is effectively a reimplementation of `GdalWarp` code - all credit for the algorithm creation goes to the GDAL developers.

As you can see, this tool is not very comprehensively documented - if you would like to add something useful
to the documentation feel free to open a PR on Github.

## Features

- `multithreading` - enables parallel functions for `Warper`. Requires `rayon`, but can provide significant
  performance improvements for some rasters.
- `io` - enables support for saving and loading `Warper` from file. Requires `rkyv`, but can be useful
  when you want to initialize `Warper` ahead-of-time.

## Example

See more usage examples in integration tests.

```rust
use mappers::{
    Ellipsoid, projections::{LambertConformalConic, LongitudeLatitude},
};
use mappers_warp::{CubicBSpline, Warper, RasterBoundsDefinition};
use ndarray::Array2;

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

let warper = Warper::initialize::<CubicBSpline, _, _>(
    &source_bounds,
    &target_bounds,
)?;

let source_raster = Array2::zeros((34, 34));
let target_raster = warper.warp_ignore_nodata(&source_raster)?;
```
