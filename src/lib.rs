use ndarray::Array2;
use thiserror::Error;

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WarperError {
    #[error("Warper has not been initiated yet. Call initiate_weights() first.")]
    NotInitiated,
    #[error("Warper has already been initiated. Call warp() to apply the transformation.")]
    AlreadyInitiated,
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

#[derive(Debug, Clone, PartialEq, PartialOrd)]
struct ResamplingKernelInternals {
    anchor_pixel_idx: [u32; 2],
    x_weights: [f64; 4],
    y_weights: [f64; 4],
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Warper {
    internals: Option<ResamplingKernelInternals>,
}

impl Warper {
    pub fn new() -> Result<Self, WarperError> {
        todo!()
    }

    pub fn initiate_weights(&mut self, kernel: impl ResamplingFilter) -> Result<(), WarperError> {
        todo!()
    }

    pub fn warp(&self, lonlat_raster: &Array2<f64>) -> Result<Array2<f64>, WarperError> {
        todo!()
    }

    #[cfg(feature = "io")]
    pub fn save_initiated(&self, path: &str) -> Result<(), WarperIOError> {
        todo!()
    }

    #[cfg(feature = "io")]
    pub fn load_initiated(path: &str) -> Result<Self, WarperIOError> {
        todo!()
    }
}
