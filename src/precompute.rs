use mappers::Projection;
use ndarray::Array2;

use crate::{
    warp_params::WarperParameters, IXJYPair, LonLatPair, RasterBounds, ResamplingFilter,
    ResamplingKernelInternals, WarperError, XYPair,
};

pub(crate) fn precompute_ixs_jys<SP: Projection, TP: Projection>(
    source_bounds: &RasterBounds<SP>,
    target_bounds: &RasterBounds<TP>,
) -> Result<Array2<IXJYPair>, WarperError> {
    let tgt_ul_edge_corner = XYPair {
        x: target_bounds.min.x - (0.5 * target_bounds.spacing.x),
        y: target_bounds.max.y + (0.5 * target_bounds.spacing.y),
    };
    let src_ul_edge_corner = LonLatPair {
        lon: source_bounds.min.x - (0.5 * source_bounds.spacing.x),
        lat: source_bounds.max.y + (0.5 * source_bounds.spacing.y),
    };

    let conversion_scaling = XYPair {
        x: 1.0 / source_bounds.spacing.x,
        y: 1.0 / source_bounds.spacing.y,
    };

    let precomputed_coords = Array2::from_shape_fn(
        (
            target_bounds.shape.j as usize,
            target_bounds.shape.i as usize,
        ),
        |(j, i)| {
            // 0.5 shift is because we are measuring from edge corner to midpoint
            let tgt_x = tgt_ul_edge_corner.x + ((i as f64 + 0.5) * target_bounds.spacing.x);
            let tgt_y = tgt_ul_edge_corner.y - ((j as f64 + 0.5) * target_bounds.spacing.y);

            let (tgt_lon, tgt_lat) = target_bounds.proj.inverse_project_unchecked(tgt_x, tgt_y);

            let result = IXJYPair {
                ix: (tgt_lon - src_ul_edge_corner.lon) * conversion_scaling.x,
                jy: (src_ul_edge_corner.lat - tgt_lat) * conversion_scaling.y,
            };

            result
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
) -> Result<Array2<ResamplingKernelInternals>, WarperError> {
    // 0.5 shift because we want to get nearest midpoint
    // but ixs, yjs are measured from the edge corner
    let internals = tgt_ixs_jys.map(|&crds| {
        let anchor_idx = (
            (crds.ix - 0.5).floor() as u32,
            (crds.jy - 0.5).floor() as u32,
        );

        let delta = compute_deltas(&crds, params);

        let x_weights = [-1, 0, 1, 2].map(|i| {
            if params.scales.x < 1.0 {
                F::apply((i as f64 - delta.x) * params.scales.x)
            } else {
                F::apply(i as f64 - delta.x)
            }
        });

        let y_weights = [-1, 0, 1, 2].map(|j| {
            if params.scales.y < 1.0 {
                F::apply((j as f64 - delta.y) * params.scales.y)
            } else {
                F::apply(j as f64 - delta.y)
            }
        });

        ResamplingKernelInternals {
            anchor_idx,
            x_weights,
            y_weights,
        }
    });

    Ok(internals)
}

#[inline(always)]
fn compute_deltas(crds: &IXJYPair, params: &WarperParameters) -> XYPair {
    let src_x = crds.ix - params.offsets.i as f64;
    let src_y = crds.jy - params.offsets.j as f64;

    let delta_x = src_x - 0.5 - (src_x - 0.5).floor();
    let delta_y = src_y - 0.5 - (src_y - 0.5).floor();

    XYPair {
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
        tests::reference_setup, warp_params::WarperParameters, CubicBSpline, IXJYPair, Warper,
    };

    use super::precompute_ixs_jys;

    #[test]
    fn ix_jy() -> Result<()> {
        let (src_bounds, tgt_bounds) = reference_setup()?;

        let ixs_jys = precompute_ixs_jys(&src_bounds, &tgt_bounds)?;

        assert_approx_eq!(f64, ixs_jys[[0, 0]].ix, 4.7102160316373727, epsilon = 1e-6);
        assert_approx_eq!(f64, ixs_jys[[0, 0]].jy, 8.8887293250701873, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn delta() -> Result<()> {
        let (src_bounds, tgt_bounds) = reference_setup()?;

        let params = WarperParameters::compute::<
            CubicBSpline,
            LongitudeLatitude,
            LambertConformalConic,
        >(&src_bounds, &tgt_bounds)?;

        let crds = IXJYPair {
            ix: 4.7102160316373727,
            jy: 8.8887293250701873,
        };

        let delta = super::compute_deltas(&crds, &params);

        assert_approx_eq!(f64, delta.x, 0.21021603163737268, epsilon = 1e-6);
        assert_approx_eq!(f64, delta.y, 0.38872932507018731, epsilon = 1e-6);

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

        for intr in warper.internals.iter() {
            let x_weights_sum = intr.x_weights.iter().sum::<f64>();
            let y_weights_sum = intr.y_weights.iter().sum::<f64>();

            assert_approx_eq!(f64, x_weights_sum, 6.0, epsilon = 1e-10);
            assert_approx_eq!(f64, y_weights_sum, 6.0, epsilon = 1e-10);
        }

        Ok(())
    }
}
