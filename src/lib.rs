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

use crate::{precompute::precompute_ixs_jys, warp_params::WarperParameters};

pub use filters::{CubicBSpline, MitchellNetravali, ResamplingFilter};
#[cfg(feature = "io")]
pub use helpers::WarperIOError;
pub(crate) use helpers::{
    GenericXYPair, IJPair, IXJYPair, MinMaxPair, RasterBounds, SourceXYPair, TargetXYPair,
};
pub use helpers::{RasterBoundsDefinition, WarperError, raster_constant_pad};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "io", derive(Serialize, Deserialize))]
struct ResamplingKernelInternals {
    pub anchor_idx: (usize, usize),
    pub x_weights: [f64; 4],
    pub y_weights: [f64; 4],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Warper {
    /// uses ndarray convention [y, x]
    source_shape: (usize, usize),
    /// internals are in a shape of target raster
    internals: Array2<ResamplingKernelInternals>,
}

/// Warper uses ndarray which implements unsafe methods.
/// From clippy: Deriving `serde::Deserialize` will create a constructor that may violate invariants held by another constructor.
/// This Wrapper prevents deriving `Deserialize` for type with usafe methods.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "io", derive(Serialize, Deserialize))]
#[cfg(feature = "io")]
struct WarperCompatIO {
    source_shape: (usize, usize),
    target_shape: (usize, usize),
    internals: Vec<ResamplingKernelInternals>,
}

#[cfg(feature = "io")]
impl From<Warper> for WarperCompatIO {
    fn from(warper_lib: Warper) -> Self {
        Self {
            source_shape: warper_lib.source_shape,
            target_shape: warper_lib.internals.dim(),
            internals: warper_lib.internals.into_flat().to_vec(),
        }
    }
}

#[cfg(feature = "io")]
impl TryFrom<WarperCompatIO> for Warper {
    type Error = ndarray::ShapeError;

    fn try_from(warper_io: WarperCompatIO) -> Result<Self, Self::Error> {
        Ok(Self {
            source_shape: warper_io.source_shape,
            internals: Array2::from_shape_vec(warper_io.target_shape, warper_io.internals)?,
        })
    }
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

    #[cfg(feature = "io")]
    pub fn save_to_file(self, path: &str) -> Result<(), WarperIOError> {
        let mut file = File::create(path)?;
        let object = WarperCompatIO::from(self);

        bincode::serde::encode_into_std_write(object, &mut file, bincode::config::standard())?;

        Ok(())
    }

    #[cfg(feature = "io")]
    pub fn load_from_file(path: &str) -> Result<Self, WarperIOError> {
        let mut file = File::open(path)?;

        let warper: WarperCompatIO =
            bincode::serde::decode_from_std_read(&mut file, bincode::config::standard())?;

        Ok(warper.try_into()?)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    #[cfg(feature = "io")]
    use crate::{Warper, filters::CubicBSpline};
    use anyhow::Result;
    use mappers::{
        Ellipsoid,
        projections::{LambertConformalConic, LongitudeLatitude},
    };
    #[cfg(feature = "io")]
    use std::fs;

    use crate::{GenericXYPair, RasterBounds, RasterBoundsDefinition};

    pub(crate) fn reference_setup_def() -> Result<(
        RasterBoundsDefinition<LongitudeLatitude>,
        RasterBoundsDefinition<LambertConformalConic>,
    )> {
        let source_projection = LongitudeLatitude;
        let target_projections = LambertConformalConic::new(
            80.,
            24.,
            12.472_955,
            35.172_804_444_444_4,
            Ellipsoid::WGS84,
        )?;

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
            target_projections,
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

    #[cfg(feature = "io")]
    #[test]
    fn io() -> Result<()> {
        let (src_bounds, tgt_bounds) = reference_setup_def()?;
        let warper = Warper::initialize::<CubicBSpline, LongitudeLatitude, LambertConformalConic>(
            &src_bounds,
            &tgt_bounds,
        )?;

        warper
            .clone()
            .save_to_file("./tests/data/saved-warper.dat")?;

        let loaded = Warper::load_from_file("./tests/data/saved-warper.dat")?;

        fs::remove_file("./tests/data/saved-warper.dat").unwrap_or(()); // cleanup

        assert_eq!(warper, loaded);

        Ok(())
    }
}
