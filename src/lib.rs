mod precompute;
mod warp_params;
mod filters;

use mappers::Projection;
use ndarray::{s, Array2};
use thiserror::Error;

#[cfg(feature = "io")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "io")]
use std::fs::File;
#[cfg(feature = "io")]
use std::io::{BufReader, BufWriter};
#[cfg(feature = "io")]
use std::path::Path;

use crate::{precompute::precompute_ixs_jys, warp_params::WarperParameters};

pub use filters::{CubicBSpline, ResamplingFilter, MitchellNetravali};

#[derive(Error, Debug)]
pub enum WarperError {
    #[error("Invalid raster dimensions")]
    InvalidRasterDimensions,

    #[error("Ndarray error {0}")]
    NdarrayError(#[from] ndarray::ShapeError),

    #[error("Projection error {0}")]
    ProjectionError(#[from] mappers::ProjectionError),

    #[error("Source raster must fully wrap")]
    SourceRasterTooSmall,

    #[error("Could not correctly convert coordinates")]
    ConversionError,

    #[error("Warping produced non-finite value")]
    WarpingError,
}

#[cfg(feature = "io")]
#[derive(Error, Debug)]
pub enum WarperIOError {
    #[error("IO error {0}")]
    IoError(#[from] std::io::Error),

    #[error("Bincode error {0}")]
    BincodeError(#[from] bincode::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub(crate) struct XYPair {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub(crate) struct LonLatPair {
    pub lon: f64,
    pub lat: f64,
}

/// Floating indexes in the source raster
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub(crate) struct IXJYPair {
    pub ix: f64,
    pub jy: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub(crate) struct IJPair {
    pub i: u32,
    pub j: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct MinMaxPair<T> {
    pub min: T,
    pub max: T,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct RasterBounds {
    pub(crate) min: XYPair,
    pub(crate) max: XYPair,
    pub(crate) spacing: XYPair,
    pub(crate) shape: IJPair,
}

impl RasterBounds {
    pub fn new(
        x_bounds: (f64, f64),
        y_bounds: (f64, f64),
        dx: f64,
        dy: f64,
    ) -> Result<Self, WarperError> {
        let (min_x, max_x) = x_bounds;
        let (min_y, max_y) = y_bounds;

        if min_x >= max_x || min_y >= max_y {
            return Err(WarperError::InvalidRasterDimensions);
        }

        let nx = (max_x - min_x) / dx;
        let ny = (max_y - min_y) / dy;

        if nx.fract() != 0.0 || ny.fract() != 0.0 {
            return Err(WarperError::InvalidRasterDimensions);
        }

        let nx = nx as u32 + 1;
        let ny = ny as u32 + 1;

        Ok(Self {
            min: XYPair { x: min_x, y: min_y },
            max: XYPair { x: max_x, y: max_y },
            spacing: XYPair { x: dx, y: dy },
            shape: IJPair { i: nx, j: ny },
        })
    }
}

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
    pub fn initialize<F: ResamplingFilter>(
        source_bounds: &RasterBounds,
        target_bounds: &RasterBounds,
        proj: &impl Projection,
    ) -> Result<Self, WarperError> {
        let params = WarperParameters::compute::<F>(source_bounds, target_bounds, proj)?;
        let tgt_ixs_jys = precompute_ixs_jys(source_bounds, target_bounds, proj)?;
        let internals = precompute::precompute_internals::<F>(&tgt_ixs_jys, &params)?;
        let source_shape = [source_bounds.shape.j, source_bounds.shape.i];

        Ok(Self {
            source_shape,
            internals,
        })
    }

    pub fn warp(&self, lonlat_raster: &Array2<f64>) -> Result<Array2<f64>, WarperError> {
        if lonlat_raster.shape()[0] != self.source_shape[0] as usize
            || lonlat_raster.shape()[1] != self.source_shape[1] as usize
        {
            return Err(WarperError::InvalidRasterDimensions);
        }

        let target_raster = self.internals.map(|intr| {
            let values = lonlat_raster.slice(s![
                (intr.anchor_idx.1 - 1) as usize..(intr.anchor_idx.1 + 3) as usize,
                (intr.anchor_idx.0 - 1) as usize..(intr.anchor_idx.0 + 3) as usize
            ]);

            let mut weight_accum = 0.0;
            let mut result_accum = 0.0;

            for j in 0..4 {
                let mut inner_weight_accum = 0.0;
                let mut inner_result_accum = 0.0;

                for i in 0..4 {
                    let value = values[[j, i]];
                    let x_weight = intr.x_weights[i];

                    inner_weight_accum += x_weight;
                    inner_result_accum += x_weight * value;
                }

                let y_weight = intr.y_weights[j];

                weight_accum += inner_weight_accum * y_weight;
                result_accum += inner_result_accum * y_weight;
            }

            result_accum / weight_accum
        });

        target_raster.fold(Ok(()), |_, &v| -> Result<(), WarperError> {
            if !v.is_finite() {
                return Err(WarperError::WarpingError);
            }

            Ok(())
        })?;

        Ok(target_raster)
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
    use crate::Warper;
    use anyhow::Result;
    use float_cmp::assert_approx_eq;
    use mappers::{projections::LambertConformalConic, Ellipsoid};
    #[cfg(feature = "io")]
    use std::fs;

    use crate::{CubicBSpline, RasterBounds, ResamplingFilter};

    pub fn reference_setup() -> Result<(RasterBounds, RasterBounds, LambertConformalConic)> {
        let source_bounds = RasterBounds::new((60.00, 67.75), (32.25, 40.0), 0.25, 0.25)?;

        let target_bounds = RasterBounds::new(
            (2_320_000. - 4_000_000., 2_740_000. - 4_000_000.),
            (5_090_000. - 4_000_000., 5_640_000. - 4_000_000.),
            10_000.,
            10_000.,
        )?;

        let proj =
            LambertConformalConic::new(80., 24., 12.472955, 35.1728044444444, Ellipsoid::WGS84)?;

        Ok((source_bounds, target_bounds, proj))
    }

    #[test]
    fn bspline_filter() {
        assert_approx_eq!(f64, CubicBSpline::apply(1.675), 0.0343281, epsilon = 1e-5);
        assert_approx_eq!(f64, CubicBSpline::apply(1.231), 0.454757, epsilon = 1e-5);
        assert_approx_eq!(f64, CubicBSpline::apply(0.115), 3.92521, epsilon = 1e-5);
        assert_approx_eq!(f64, CubicBSpline::apply(-0.243), 3.68875, epsilon = 1e-5);
        assert_approx_eq!(f64, CubicBSpline::apply(-1.65), 0.042875, epsilon = 1e-5);
    }

    #[cfg(feature = "io")]
    #[test]
    fn io() -> Result<()> {
        let (src_bounds, tgt_bounds, proj) = reference_setup()?;
        let warper = Warper::initialize::<CubicBSpline>(&src_bounds, &tgt_bounds, &proj)?;

        warper.save_to_file("./tests/data/saved-warper.dat")?;

        let loaded = Warper::load_from_file("./tests/data/saved-warper.dat")?;

        fs::remove_file("./tests/data/saved-warper.dat").unwrap_or(()); // cleanup

        assert_eq!(warper, loaded);

        Ok(())
    }
}
