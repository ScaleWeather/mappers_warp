use ndarray::{s, Array2, FoldWhile, Zip};

use crate::{Warper, WarperError};

impl Warper {
    #[must_use]
    pub fn warp_unchecked(&self, source_raster: &Array2<f64>) -> Array2<f64> {
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

        target_raster
    }

    // From GdalWarp: for bilinear, cubic, cubicspline and lanczos, for each target pixel, the coordinate of its center
    // is projected back to source coordinates and a corresponding source pixel is identified. If this source pixel is invalid,
    // the target pixel is considered as nodata. Given that those resampling kernels have a non-null kernel radius,
    // this source pixel is just one among other several source pixels, and it might be possible that there are invalid
    // values in those other contributing source pixels. The weights used to take into account those invalid values
    // will be set to zero to ignore them.
    pub fn warp_ignore_nodata(
        &self,
        source_raster: &Array2<f64>,
    ) -> Result<Array2<f64>, WarperError> {
        if source_raster.shape()[0] != self.source_shape[0] as usize
            || source_raster.shape()[1] != self.source_shape[1] as usize
        {
            return Err(WarperError::InvalidRasterDimensions);
        }

        let mut target_raster = Array2::from_elem(self.internals.raw_dim(), f64::NEG_INFINITY);

        Zip::from(&mut target_raster)
            .and(&self.internals)
            .fold_while(Ok(()), |_, v, intr| {
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

                        if !value.is_nan() {
                            let x_weight = intr.x_weights[i];
                            inner_weight_accum += x_weight;
                            inner_result_accum += x_weight * value;
                        }
                    }

                    let y_weight = intr.y_weights[j];

                    weight_accum += inner_weight_accum * y_weight;
                    result_accum += inner_result_accum * y_weight;
                }

                if (weight_accum - 0.0).abs() < f64::EPSILON {
                    *v = f64::NAN;
                    return FoldWhile::Continue(Ok(()));
                }

                let result = result_accum / weight_accum;

                if result.is_finite() {
                    *v = result;
                    FoldWhile::Continue(Ok(()))
                } else {
                    FoldWhile::Done(Err(WarperError::WarpingError))
                }
            })
            .into_inner()?;

        Ok(target_raster)
    }

    pub fn warp_reject_nodata(
        &self,
        source_raster: &Array2<f64>,
    ) -> Result<Array2<f64>, WarperError> {
        if source_raster.shape()[0] != self.source_shape[0] as usize
            || source_raster.shape()[1] != self.source_shape[1] as usize
        {
            return Err(WarperError::InvalidRasterDimensions);
        }

        let mut target_raster = Array2::from_elem(self.internals.raw_dim(), f64::NEG_INFINITY);

        Zip::from(&mut target_raster)
            .and(&self.internals)
            .fold_while(Ok(()), |_, v, intr| {
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

                        if value.is_nan() {
                            return FoldWhile::Done(Err(WarperError::WarpingError));
                        }
                        let x_weight = intr.x_weights[i];
                        inner_weight_accum += x_weight;
                        inner_result_accum += x_weight * value;
                    }

                    let y_weight = intr.y_weights[j];

                    weight_accum += inner_weight_accum * y_weight;
                    result_accum += inner_result_accum * y_weight;
                }

                let result = result_accum / weight_accum;

                if result.is_finite() {
                    *v = result;
                    FoldWhile::Continue(Ok(()))
                } else {
                    FoldWhile::Done(Err(WarperError::WarpingError))
                }
            })
            .into_inner()?;

        Ok(target_raster)
    }

    pub fn warp_discard_nodata(
        &self,
        source_raster: &Array2<f64>,
    ) -> Result<Array2<f64>, WarperError> {
        if source_raster.shape()[0] != self.source_shape[0] as usize
            || source_raster.shape()[1] != self.source_shape[1] as usize
        {
            return Err(WarperError::InvalidRasterDimensions);
        }

        let mut target_raster = Array2::from_elem(self.internals.raw_dim(), f64::NEG_INFINITY);

        Zip::from(&mut target_raster)
            .and(&self.internals)
            .fold_while(Ok(()), |_, v, intr| {
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

                        if value.is_nan() {
                            *v = f64::NAN;
                            return FoldWhile::Continue(Ok(()));
                        }
                        let x_weight = intr.x_weights[i];
                        inner_weight_accum += x_weight;
                        inner_result_accum += x_weight * value;
                    }

                    let y_weight = intr.y_weights[j];

                    weight_accum += inner_weight_accum * y_weight;
                    result_accum += inner_result_accum * y_weight;
                }

                let result = result_accum / weight_accum;

                if result.is_finite() {
                    *v = result;
                    FoldWhile::Continue(Ok(()))
                } else {
                    FoldWhile::Done(Err(WarperError::WarpingError))
                }
            })
            .into_inner()?;

        Ok(target_raster)
    }
}
