mod precompute;
mod warp_params;

use mappers::Projection;
use ndarray::{s, Array2};
use thiserror::Error;

use crate::{precompute::precompute_ixs_jys, warp_params::WarperParameters};

#[derive(Error, Debug)]
pub enum WarperError {
    #[error("Invalid raster dimensions.")]
    InvalidRasterDimensions,

    #[error("Ndarray error.")]
    NdarrayError(#[from] ndarray::ShapeError),

    #[error("Projection error.")]
    ProjectionError(#[from] mappers::ProjectionError),

    #[error("Source raster must fully wrap.")]
    SourceRasterTooSmall,

    #[error("Could not correctly convert coordinates.")]
    ConversionError,

    #[error("Warping produced non-finite value.")]
    WarpingError,
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
        let xp2 = x + 2.0;
        let xp1 = x + 1.0;
        let xm1 = x - 1.0;

        let xp2c = xp2 * xp2 * xp2;

        let mut res = 0.0;

        if xm1 > 0.0 {
            res += -4.0 * xm1 * xm1 * xm1;
        };

        if x > 0.0 {
            res += 6.0 * x * x * x;
        };

        if xp1 > 0.0 {
            res += -4.0 * xp1 * xp1 * xp1;
        };

        if xp2 > 0.0 {
            res += xp2c;
        };

        res
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
    pub anchor_idx: (u32, u32),
    pub x_weights: [f64; 4],
    pub y_weights: [f64; 4],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Warper {
    source_shape: [u32; 2],
    internals: Array2<ResamplingKernelInternals>,
}

impl Warper {
    pub fn initialize<P: Projection, F: ResamplingFilter>(
        source_bounds: &RasterBounds,
        target_bounds: &RasterBounds,
        proj: &P,
    ) -> Result<Self, WarperError> {
        let params = WarperParameters::compute::<F>(source_bounds, target_bounds, proj)?;
        let tgt_ixs_jys = precompute_ixs_jys(source_bounds, target_bounds, proj)?;
        let internals = precompute::precompute_internals::<F>(&tgt_ixs_jys, &params)?;
        let source_shape = [source_bounds.shape.i, source_bounds.shape.j];

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
                (intr.anchor_idx.0 - 1) as usize..(intr.anchor_idx.0 + 3) as usize,
                (intr.anchor_idx.1 - 1) as usize..(intr.anchor_idx.1 + 3) as usize
            ]);

            let mut weight_accum = 0.0;
            let mut result_accum = 0.0;

            for j in 0..4 {
                let mut inner_weight_accum = 0.0;
                let mut inner_result_accum = 0.0;

                for i in 0..4 {
                    let value = values[[i, j]];
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
        todo!()
    }

    #[cfg(feature = "io")]
    pub fn load_from_file(path: &str) -> Result<Self, WarperIOError> {
        todo!()
    }
}

#[cfg(test)]
pub mod tests {
    use float_cmp::assert_approx_eq;
    use mappers::{projections::LambertConformalConic, Ellipsoid};

    use crate::{CubicBSpline, RasterBounds, ResamplingFilter, Warper};

    pub fn reference_setup() -> (RasterBounds, RasterBounds, LambertConformalConic) {
        let source_bounds = RasterBounds::new((60.00, 67.75), (32.25, 40.0), 0.25, 0.25).unwrap();

        let target_bounds = RasterBounds::new(
            (2_320_000. - 4_000_000., 2_740_000. - 4_000_000.),
            (5_090_000. - 4_000_000., 5_640_000. - 4_000_000.),
            10_000.,
            10_000.,
        )
        .unwrap();

        let proj =
            LambertConformalConic::new(80., 24., 12.472955, 35.1728044444444, Ellipsoid::WGS84)
                .unwrap();

        return (source_bounds, target_bounds, proj);
    }

    #[test]
    fn internals() {
        let (src_bounds, tgt_bounds, proj) = reference_setup();

        let warper = Warper::initialize::<LambertConformalConic, CubicBSpline>(
            &src_bounds,
            &tgt_bounds,
            &proj,
        )
        .unwrap();

        assert_eq!(warper.internals[[0, 0]].anchor_idx, (4, 8));

        for intr in warper.internals.iter() {
            let x_weights_sum = intr.x_weights.iter().sum::<f64>();
            let y_weights_sum = intr.y_weights.iter().sum::<f64>();

            assert_approx_eq!(f64, x_weights_sum, 6.0, epsilon = 1e-10);
            assert_approx_eq!(f64, y_weights_sum, 6.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn bspline_filter() {
        assert_approx_eq!(f64, CubicBSpline::apply(1.675), 0.0343281, epsilon = 1e-5);
        assert_approx_eq!(f64, CubicBSpline::apply(1.231), 0.454757, epsilon = 1e-5);
        assert_approx_eq!(f64, CubicBSpline::apply(0.115), 3.92521, epsilon = 1e-5);
        assert_approx_eq!(f64, CubicBSpline::apply(-0.243), 3.68875, epsilon = 1e-5);
        assert_approx_eq!(f64, CubicBSpline::apply(-1.65), 0.042875, epsilon = 1e-5);
    }
}
