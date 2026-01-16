//! # Mappers Warp
//!
//! Very simplistic tool for reprojecting maps, based on the `GdalWarp`, using mappers for geographic projection.
//!
//! This tool is effectively a reimplementation of `GdalWarp` code - all credit for the algorithm creation goes to the GDAL developers.
//!
//! As you can see, this tool is not very comprehensively documented - if you would like to add something useful
//! to the documentation feel free to open a PR on Github.
//!
//! ## Features
//!
//! - `multithreading` - enables parallel functions for `Warper`. Requires `rayon`, but can provide significant
//!   performance improvements for some rasters.
//! - `io` - enables support for saving and loading `Warper` from file. Requires `rkyv`, but can be useful
//!   when you want to initialize `Warper` ahead-of-time.
//!
//! ## Example
//!
//! See more usage examples in integration tests.
//!
//! ```
//! use mappers::{
//!     Ellipsoid, projections::{LambertConformalConic, LongitudeLatitude},
//! };
//! use mappers_warp::{CubicBSpline, Warper, RasterBoundsDefinition};
//! use ndarray::Array2;
//! # fn main() -> anyhow::Result<()> {
//! let src_proj = LongitudeLatitude;
//! let tgt_proj = LambertConformalConic::builder()
//!     .ref_lonlat(80., 24.)
//!     .standard_parallels(12.472955, 35.1728044444444)
//!     .ellipsoid(Ellipsoid::WGS84)
//!     .initialize_projection()?;
//!
//! let source_bounds =
//!     RasterBoundsDefinition::new((60.00, 68.25), (31.75, 40.0), 0.25, 0.25, src_proj)?;
//! let target_bounds = RasterBoundsDefinition::new(
//!     (2_320_000. - 4_000_000., 2_740_000. - 4_000_000.),
//!     (5_090_000. - 4_000_000., 5_640_000. - 4_000_000.),
//!     10_000.,
//!     10_000.,
//!     tgt_proj,
//! )?;
//!
//! let warper = Warper::initialize::<CubicBSpline, _, _>(
//!     &source_bounds,
//!     &target_bounds,
//! )?;
//!
//! let source_raster = Array2::zeros((34, 34));
//! let target_raster = warper.warp_ignore_nodata(&source_raster)?;
//! # Ok(())
//! # }
//! ```

#![warn(clippy::pedantic)]
#![warn(clippy::perf)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::excessive_precision)]
#![cfg_attr(docsrs, feature(doc_cfg))]

mod compute;
mod filters;
mod helpers;
mod precompute;
mod warp_params;

#[cfg(feature = "io")]
#[cfg_attr(docsrs, doc(cfg(feature = "io")))]
mod io;

use std::fmt::Debug;

pub use filters::{CubicBSpline, MitchellNetravali, ResamplingFilter};
#[cfg(feature = "io")]
pub use helpers::WarperIOError;
pub(crate) use helpers::{
    GenericXYPair, IJPair, IXJYPair, MinMaxPair, RasterBounds, SourceXYPair, TargetXYPair,
};
pub use helpers::{RasterBoundsDefinition, WarperError, raster_constant_pad};

use mappers::Projection;
use ndarray::Array2;
#[cfg(feature = "io")]
use rkyv::{Archive, Deserialize, Serialize};

use crate::{precompute::precompute_ixs_jys, warp_params::WarperParameters};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "io", derive(Archive, Deserialize, Serialize))]
struct ResamplingKernelInternals {
    pub anchor_idx: (usize, usize),
    pub x_weights: [f64; 4],
    pub y_weights: [f64; 4],
}

/// Main struct used for the warp operation
#[derive(Debug, Clone, PartialEq)]
pub struct Warper {
    /// uses ndarray convention [y, x]
    source_shape: (usize, usize),
    /// internals are in a shape of target raster
    internals: Array2<ResamplingKernelInternals>,
}

impl Warper {
    pub fn initialize<F: ResamplingFilter, SP: Projection, TP: Projection>(
        source_bounds: &RasterBoundsDefinition<SP>,
        target_bounds: &RasterBoundsDefinition<TP>,
    ) -> Result<Self, WarperError> {
        let source_bounds =
            RasterBounds::<SP, GenericXYPair>::from(source_bounds).cast_xy_pairs::<SourceXYPair>();
        let target_bounds =
            RasterBounds::<TP, GenericXYPair>::from(target_bounds).cast_xy_pairs::<TargetXYPair>();

        let params = WarperParameters::compute::<F, SP, TP>(&source_bounds, &target_bounds)?;
        let tgt_ixs_jys = precompute_ixs_jys(&source_bounds, &target_bounds)?;
        let internals = precompute::precompute_internals::<F>(&tgt_ixs_jys, &params);
        let source_shape = (
            source_bounds.shape.j as usize,
            source_bounds.shape.i as usize,
        );

        Ok(Self {
            source_shape,
            internals,
        })
    }

    #[cfg(feature = "multithreading")]
    #[cfg_attr(docsrs, doc(cfg(feature = "multithreading")))]
    /// Same as initialize but uses multithreading in some computations.
    pub fn initialize_parallel<F: ResamplingFilter, SP: Projection, TP: Projection>(
        source_bounds: &RasterBoundsDefinition<SP>,
        target_bounds: &RasterBoundsDefinition<TP>,
    ) -> Result<Self, WarperError> {
        use crate::precompute::precompute_ixs_jys_parallel;

        let source_bounds =
            RasterBounds::<SP, GenericXYPair>::from(source_bounds).cast_xy_pairs::<SourceXYPair>();
        let target_bounds =
            RasterBounds::<TP, GenericXYPair>::from(target_bounds).cast_xy_pairs::<TargetXYPair>();

        let params = WarperParameters::compute::<F, SP, TP>(&source_bounds, &target_bounds)?;
        let tgt_ixs_jys = precompute_ixs_jys_parallel(&source_bounds, &target_bounds)?;
        let internals = precompute::precompute_internals_parallel::<F>(&tgt_ixs_jys, &params);
        let source_shape = (
            source_bounds.shape.j as usize,
            source_bounds.shape.i as usize,
        );

        Ok(Self {
            source_shape,
            internals,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use anyhow::Result;
    use mappers::{
        Ellipsoid,
        projections::{LambertConformalConic, LongitudeLatitude},
    };

    use crate::{GenericXYPair, RasterBounds, RasterBoundsDefinition};

    pub(crate) fn reference_setup_def() -> Result<(
        RasterBoundsDefinition<LongitudeLatitude>,
        RasterBoundsDefinition<LambertConformalConic>,
    )> {
        let source_projection = LongitudeLatitude;
        let target_projection = LambertConformalConic::builder()
            .ref_lonlat(80., 24.)
            .standard_parallels(12.472_955, 35.172_804_444_444_4)
            .ellipsoid(Ellipsoid::WGS84)
            .initialize_projection()?;

        let source_bounds = RasterBoundsDefinition::new(
            (60.00, 67.75),
            (32.25, 40.0),
            0.25,
            0.25,
            source_projection,
        )?;

        let target_bounds = RasterBoundsDefinition::new(
            (2_320_000. - 4_000_000., 2_740_000. - 4_000_000.),
            (5_090_000. - 4_000_000., 5_640_000. - 4_000_000.),
            10_000.,
            10_000.,
            target_projection,
        )?;

        Ok((source_bounds, target_bounds))
    }

    pub(crate) fn reference_setup() -> Result<(
        RasterBounds<LongitudeLatitude, GenericXYPair>,
        RasterBounds<LambertConformalConic, GenericXYPair>,
    )> {
        let (source_bounds, target_bounds) = reference_setup_def()?;
        Ok((source_bounds.into(), target_bounds.into()))
    }
}
