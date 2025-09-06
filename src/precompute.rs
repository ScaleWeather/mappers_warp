use mappers::Projection;
use ndarray::Array2;

use crate::{
    helpers::GenericXYPair, warp_params::WarperParameters, IXJYPair, RasterBounds,
    ResamplingFilter, ResamplingKernelInternals, SourceXYPair, TargetXYPair, WarperError,
};

pub(crate) fn precompute_ixs_jys<SP: Projection, TP: Projection>(
    source_bounds: &RasterBounds<SP, SourceXYPair>,
    target_bounds: &RasterBounds<TP, TargetXYPair>,
) -> Result<Array2<IXJYPair>, WarperError> {
    let tgt_ul_edge_corner = SourceXYPair {
        x: target_bounds.min.x - (0.5 * target_bounds.spacing.x),
        y: target_bounds.max.y + (0.5 * target_bounds.spacing.y),
    };
    let src_ul_edge_corner = SourceXYPair {
        x: source_bounds.min.x - (0.5 * source_bounds.spacing.x),
        y: source_bounds.max.y + (0.5 * source_bounds.spacing.y),
    };

    let conversion_scaling = GenericXYPair {
        x: 1.0 / source_bounds.spacing.x,
        y: 1.0 / source_bounds.spacing.y,
    };

    let proj_pipe = &target_bounds.proj.pipe_to(&source_bounds.proj);

    let precomputed_coords = Array2::from_shape_fn(
        (
            target_bounds.shape.j as usize,
            target_bounds.shape.i as usize,
        ),
        |(j, i)| {
            // 0.5 shift is because we are measuring from edge corner to midpoint
            let tgt_x = tgt_ul_edge_corner.x + ((i as f64 + 0.5) * target_bounds.spacing.x);
            let tgt_y = tgt_ul_edge_corner.y - ((j as f64 + 0.5) * target_bounds.spacing.y);

            let (src_x, src_y) = proj_pipe.convert_unchecked(tgt_x, tgt_y);

            IXJYPair {
                ix: (src_x - src_ul_edge_corner.x) * conversion_scaling.x,
                jy: (src_ul_edge_corner.y - src_y) * conversion_scaling.y,
            }
        },
    );

    precomputed_coords.fold(Ok(()), |_, &v| -> Result<(), WarperError> {
        if !v.ix.is_finite() || !v.jy.is_finite() {
            return Err(WarperError::ConversionError);
        }

        Ok(())
    })?;

    Ok(precomputed_coords)
}

pub(crate) fn precompute_internals<F: ResamplingFilter>(
    tgt_ixs_jys: &Array2<IXJYPair>,
    params: &WarperParameters,
) -> Array2<ResamplingKernelInternals> {
    // 0.5 shift because we want to get nearest midpoint
    // but ixs, yjs are measured from the edge corner
    tgt_ixs_jys.map(|&crds| {
        let anchor_idx = (
            (crds.ix - 0.5).floor() as usize,
            (crds.jy - 0.5).floor() as usize,
        );

        let delta = compute_deltas(&crds, params);

        let x_weights = [-1, 0, 1, 2].map(|i| {
            if params.scales.x < 1.0 {
                F::apply((f64::from(i) - delta.x) * params.scales.x)
            } else {
                F::apply(f64::from(i) - delta.x)
            }
        });

        let y_weights = [-1, 0, 1, 2].map(|j| {
            if params.scales.y < 1.0 {
                F::apply((f64::from(j) - delta.y) * params.scales.y)
            } else {
                F::apply(f64::from(j) - delta.y)
            }
        });

        ResamplingKernelInternals {
            anchor_idx,
            x_weights,
            y_weights,
        }
    })
}

#[inline]
fn compute_deltas(crds: &IXJYPair, params: &WarperParameters) -> GenericXYPair {
    let src_x = crds.ix - f64::from(params.offsets.i);
    let src_y = crds.jy - f64::from(params.offsets.j);

    let delta_x = src_x - 0.5 - (src_x - 0.5).floor();
    let delta_y = src_y - 0.5 - (src_y - 0.5).floor();

    GenericXYPair {
        x: delta_x,
        y: delta_y,
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use float_cmp::assert_approx_eq;
    use mappers::projections::{LambertConformalConic, LongitudeLatitude};

    use crate::{
        tests::reference_setup, warp_params::WarperParameters, CubicBSpline, IXJYPair,
        SourceXYPair, TargetXYPair, Warper,
    };

    use super::precompute_ixs_jys;

    #[test]
    fn ix_jy() -> Result<()> {
        let (src_bounds, tgt_bounds) = reference_setup()?;

        let src_bounds = src_bounds.cast_xy_pairs::<SourceXYPair>();
        let tgt_bounds = tgt_bounds.cast_xy_pairs::<TargetXYPair>();

        let ixs_jys = precompute_ixs_jys(&src_bounds, &tgt_bounds)?;

        assert_approx_eq!(
            f64,
            ixs_jys[[0, 0]].ix,
            4.710_216_031_637_372_7,
            epsilon = 1e-6
        );
        assert_approx_eq!(
            f64,
            ixs_jys[[0, 0]].jy,
            8.888_729_325_070_187_3,
            epsilon = 1e-6
        );

        Ok(())
    }

    #[test]
    fn delta() -> Result<()> {
        let (src_bounds, tgt_bounds) = reference_setup()?;

        let src_bounds = src_bounds.cast_xy_pairs::<SourceXYPair>();
        let tgt_bounds = tgt_bounds.cast_xy_pairs::<TargetXYPair>();

        let params = WarperParameters::compute::<
            CubicBSpline,
            LongitudeLatitude,
            LambertConformalConic,
        >(&src_bounds, &tgt_bounds)?;

        let crds = IXJYPair {
            ix: 4.710_216_031_637_372_7,
            jy: 8.888_729_325_070_187_3,
        };

        let delta = super::compute_deltas(&crds, &params);

        assert_approx_eq!(f64, delta.x, 0.210_216_031_637_372_68, epsilon = 1e-6);
        assert_approx_eq!(f64, delta.y, 0.388_729_325_070_187_31, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn internals() -> Result<()> {
        let (src_bounds, tgt_bounds) = reference_setup()?;

        let warper = Warper::initialize::<CubicBSpline, LongitudeLatitude, LambertConformalConic>(
            &src_bounds,
            &tgt_bounds,
        )?;

        assert_eq!(warper.internals[[0, 0]].anchor_idx, (4, 8));

        for intr in &warper.internals {
            let x_weights_sum = intr.x_weights.iter().sum::<f64>();
            let y_weights_sum = intr.y_weights.iter().sum::<f64>();

            assert_approx_eq!(f64, x_weights_sum, 6.0, epsilon = 1e-10);
            assert_approx_eq!(f64, y_weights_sum, 6.0, epsilon = 1e-10);
        }

        Ok(())
    }
}
