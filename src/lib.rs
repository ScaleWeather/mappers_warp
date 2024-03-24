use ndarray::Array2;
use thiserror::Error;

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WarperError {
    #[error("Invalid raster dimensions.")]
    InvalidRasterDimensions,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CubicBSpline;

impl ResamplingFilter for CubicBSpline {
    fn apply(x: f64) -> f64 {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MitchellNetravali;

impl ResamplingFilter for MitchellNetravali {
    fn apply(x: f64) -> f64 {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct XYTuple<T> {
    pub x: T,
    pub y: T,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct RasterBounds {
    pub min: XYTuple<f64>,
    pub max: XYTuple<f64>,
    pub spacing: XYTuple<f64>,
    pub shape: XYTuple<u32>,
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
            min: XYTuple { x: min_x, y: min_y },
            max: XYTuple { x: max_x, y: max_y },
            spacing: XYTuple { x: dx, y: dy },
            shape: XYTuple { x: nx, y: ny },
        })
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
struct ResamplingKernelInternals {
    anchor_pixel_idx: [u32; 2],
    x_weights: [f64; 4],
    y_weights: [f64; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WarperBuilder {}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Warper {
    internals: ResamplingKernelInternals,
}

impl WarperBuilder {
    pub fn new(
        source_bounds: &RasterBounds,
        target_bounds: &RasterBounds,
    ) -> Result<Self, WarperError> {
        todo!()
    }

    pub fn initiate_weights(&self, kernel: impl ResamplingFilter) -> Result<Warper, WarperError> {
        todo!()
    }
}
impl Warper {
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
