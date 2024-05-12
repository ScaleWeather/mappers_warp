mod filters;
mod helpers;
mod precompute;
mod warp_params;

use mappers::Projection;
use ndarray::{s, Array2};

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
pub use helpers::{GenericXYPair, RasterBounds, WarperError};
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
        let internals = precompute::precompute_internals::<F>(&tgt_ixs_jys, &params)?;
        let source_shape = [source_bounds.shape.j, source_bounds.shape.i];

        Ok(Self {
            source_shape,
            internals,
        })
    }

    // From GdalWarp: for bilinear, cubic, cubicspline and lanczos, for each target pixel, the coordinate of its center
    // is projected back to source coordinates and a corresponding source pixel is identified. If this source pixel is invalid,
    // the target pixel is considered as nodata. Given that those resampling kernels have a non-null kernel radius,
    // this source pixel is just one among other several source pixels, and it might be possible that there are invalid
    // values in those other contributing source pixels. The weights used to take into account those invalid values
    // will be set to zero to ignore them.
    pub fn warp(&self, source_raster: &Array2<f64>) -> Result<Array2<f64>, WarperError> {
        if source_raster.shape()[0] != self.source_shape[0] as usize
            || source_raster.shape()[1] != self.source_shape[1] as usize
        {
            return Err(WarperError::InvalidRasterDimensions);
        }

        let target_raster = self.internals.map(|intr| {
            let values = source_raster.slice(s![
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
        let target_projections =
            LambertConformalConic::new(80., 24., 12.472955, 35.1728044444444, Ellipsoid::WGS84)?;

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
