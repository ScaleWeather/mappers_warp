//! # Mappers Warp
//!
//! Very simplistic tool for reprojecting maps, based on the `GdalWarp`, using mappers for geographic projection.
//!
//! This tool is effectively a reimplementation of `GdalWarp` code - all credit for the algorithm creation goes to the GDAL developers.
//!
//! Unfortunately, there is no documentation for this crate. If you would like to add the docs, feel free to create a Pull Request on Github.

#![warn(clippy::pedantic)]
#![warn(clippy::perf)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::excessive_precision)]

mod compute;
mod filters;
mod helpers;
mod precompute;
mod warp_params;

use mappers::Projection;
use ndarray::Array2;

#[cfg(feature = "io")]
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
#[cfg(feature = "io")]
use std::fs::File;
#[cfg(feature = "io")]
use std::io::{BufReader, BufWriter};
#[cfg(feature = "io")]
use std::path::Path;

use crate::{precompute::precompute_ixs_jys, warp_params::WarperParameters};

pub use filters::{CubicBSpline, MitchellNetravali, ResamplingFilter};
#[cfg(feature = "io")]
pub use helpers::WarperIOError;
pub use helpers::{raster_constant_pad, GenericXYPair, RasterBounds, WarperError};
pub(crate) use helpers::{IJPair, IXJYPair, MinMaxPair, SourceXYPair, TargetXYPair};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "io", derive(Serialize, Deserialize))]
struct ResamplingKernelInternals {
    pub anchor_idx: (u32, u32),
    pub x_weights: [f64; 4],
    pub y_weights: [f64; 4],
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "io", derive(Serialize, Deserialize))]
pub struct Warper {
    // uses ndarray convention [y, x]
    source_shape: [u32; 2],
    internals: Array2<ResamplingKernelInternals>,
}

impl Warper {
    pub fn initialize<F: ResamplingFilter, SP: Projection, TP: Projection>(
        source_bounds: &RasterBounds<SP, GenericXYPair>,
        target_bounds: &RasterBounds<TP, GenericXYPair>,
    ) -> Result<Self, WarperError> {
        let source_bounds = &source_bounds.cast_xy_pairs::<SourceXYPair>();
        let target_bounds = &target_bounds.cast_xy_pairs::<TargetXYPair>();

        let params = WarperParameters::compute::<F, SP, TP>(source_bounds, target_bounds)?;
        let tgt_ixs_jys = precompute_ixs_jys(source_bounds, target_bounds)?;
        let internals = precompute::precompute_internals::<F>(&tgt_ixs_jys, &params);
        let source_shape = [source_bounds.shape.j, source_bounds.shape.i];

        Ok(Self {
            source_shape,
            internals,
        })
    }

    #[cfg(feature = "io")]
    pub fn save_to_file(&self, path: &str) -> Result<(), WarperIOError> {
        let path = Path::new(path);
        let file = File::create(path)?;
        let mut buf = BufWriter::new(file);

        bincode::serialize_into(&mut buf, &self)?;

        Ok(())
    }

    #[cfg(feature = "io")]
    pub fn load_from_file(path: &str) -> Result<Self, WarperIOError> {
        let path = Path::new(path);
        let file = File::open(path)?;
        let mut buf = BufReader::new(file);

        let warper: Self = bincode::deserialize_from(&mut buf)?;

        Ok(warper)
    }
}

#[cfg(test)]
pub mod tests {
    #[cfg(feature = "io")]
    use crate::{filters::CubicBSpline, Warper};
    use anyhow::Result;
    use mappers::{
        projections::{LambertConformalConic, LongitudeLatitude},
        Ellipsoid,
    };
    #[cfg(feature = "io")]
    use std::fs;

    use crate::{GenericXYPair, RasterBounds};

    pub fn reference_setup() -> Result<(
        RasterBounds<LongitudeLatitude, GenericXYPair>,
        RasterBounds<LambertConformalConic, GenericXYPair>,
    )> {
        let source_projection = LongitudeLatitude;
        let target_projections = LambertConformalConic::new(
            80.,
            24.,
            12.472_955,
            35.172_804_444_444_4,
            Ellipsoid::WGS84,
        )?;

        let source_bounds =
            RasterBounds::new((60.00, 67.75), (32.25, 40.0), 0.25, 0.25, source_projection)?;

        let target_bounds = RasterBounds::new(
            (2_320_000. - 4_000_000., 2_740_000. - 4_000_000.),
            (5_090_000. - 4_000_000., 5_640_000. - 4_000_000.),
            10_000.,
            10_000.,
            target_projections,
        )?;

        Ok((source_bounds, target_bounds))
    }

    #[cfg(feature = "io")]
    #[test]
    fn io() -> Result<()> {
        let (src_bounds, tgt_bounds) = reference_setup()?;
        let warper = Warper::initialize::<CubicBSpline, LongitudeLatitude, LambertConformalConic>(
            &src_bounds,
            &tgt_bounds,
        )?;

        warper.save_to_file("./tests/data/saved-warper.dat")?;

        let loaded = Warper::load_from_file("./tests/data/saved-warper.dat")?;

        fs::remove_file("./tests/data/saved-warper.dat").unwrap_or(()); // cleanup

        assert_eq!(warper, loaded);

        Ok(())
    }
}
