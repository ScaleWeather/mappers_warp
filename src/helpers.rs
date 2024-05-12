use mappers::Projection;
use std::fmt::Debug;
use thiserror::Error;

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

pub trait XYPair: Debug + Clone + Copy + PartialEq + PartialOrd {}
impl XYPair for GenericXYPair {}
impl XYPair for SourceXYPair {}
impl XYPair for TargetXYPair {}

impl From<GenericXYPair> for SourceXYPair {
    fn from(v: GenericXYPair) -> Self {
        SourceXYPair { x: v.x, y: v.y }
    }
}

impl From<GenericXYPair> for TargetXYPair {
    fn from(v: GenericXYPair) -> Self {
        TargetXYPair { x: v.x, y: v.y }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct GenericXYPair {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub(crate) struct SourceXYPair {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub(crate) struct TargetXYPair {
    pub x: f64,
    pub y: f64,
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
pub struct RasterBounds<P: Projection, T: XYPair> {
    pub(crate) min: T,
    pub(crate) max: T,
    pub(crate) spacing: GenericXYPair,
    pub(crate) shape: IJPair,
    pub(crate) proj: P,
}

impl<P: Projection> RasterBounds<P, GenericXYPair> {
    pub fn new(
        x_bounds: (f64, f64),
        y_bounds: (f64, f64),
        dx: f64,
        dy: f64,
        proj: P,
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
            min: GenericXYPair { x: min_x, y: min_y },
            max: GenericXYPair { x: max_x, y: max_y },
            spacing: GenericXYPair { x: dx, y: dy },
            shape: IJPair { i: nx, j: ny },
            proj,
        })
    }

    pub(crate) fn cast_xy_pairs<T: From<GenericXYPair> + XYPair>(&self) -> RasterBounds<P, T> {
        RasterBounds {
            min: T::from(self.min),
            max: T::from(self.max),
            spacing: self.spacing,
            shape: self.shape,
            proj: self.proj,
        }
    }
}
