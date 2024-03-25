mod warp_params;

use mappers::Projection;
use ndarray::Array2;
use thiserror::Error;

use crate::warp_params::WarperParameters;

#[derive(Error, Debug)]
pub enum WarperError {
    #[error("Invalid raster dimensions.")]
    InvalidRasterDimensions,

    #[error("Ndarray error.")]
    NdarrayError(#[from] ndarray::ShapeError),

    #[error("Projection error.")]
    ProjectionError(#[from] mappers::ProjectionError),

    #[error("Source raster must fully wrap .")]
    SourceRasterTooSmall,
}

#[cfg(feature = "io")]
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WarperIOError {
    #[error("File not found.")]
    FileNotFound,

    #[error("Invalid file.")]
    InvalidFile,
}

pub trait ResamplingFilter {
    fn apply(x: f64) -> f64;

    const X_RADIUS: f64;
    const Y_RADIUS: f64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CubicBSpline;

impl ResamplingFilter for CubicBSpline {
    fn apply(x: f64) -> f64 {
        todo!()
    }

    const X_RADIUS: f64 = 2.0;
    const Y_RADIUS: f64 = 2.0;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MitchellNetravali;

impl ResamplingFilter for MitchellNetravali {
    fn apply(x: f64) -> f64 {
        todo!()
    }

    const X_RADIUS: f64 = 2.0;
    const Y_RADIUS: f64 = 2.0;
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
struct ResamplingKernelInternals {
    anchor_pixel_idx: [u32; 2],
    x_weights: [f64; 4],
    y_weights: [f64; 4],
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Warper {
    internals: ResamplingKernelInternals,
}

impl Warper {
    pub fn initialize<P: Projection, F: ResamplingFilter>(
        source_bounds: &RasterBounds,
        target_bounds: &RasterBounds,
        proj: &P,
        kernel: &F,
    ) -> Result<Self, WarperError> {
        let params = WarperParameters::compute::<F>(source_bounds, target_bounds, proj)?;

        todo!()
    }

    pub fn warp(&self, lonlat_raster: &Array2<f64>) -> Result<Array2<f64>, WarperError> {
        todo!()
    }

    #[cfg(feature = "io")]
    pub fn save_to_file(&self, path: &str) -> Result<(), WarperIOError> {
        todo!()
    }

    #[cfg(feature = "io")]
    pub fn load_from_file(path: &str) -> Result<Self, WarperIOError> {
        todo!()
    }
}
