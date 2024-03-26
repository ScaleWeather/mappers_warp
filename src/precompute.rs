use mappers::Projection;
use ndarray::Array2;

use crate::{
    warp_params::WarperParameters, IXJYPair, LonLatPair, RasterBounds, ResamplingFilter,
    ResamplingKernelInternals, WarperError, XYPair,
};

// DEBUG: This function is okay
pub(crate) fn precompute_ixs_jys(
    source_bounds: &RasterBounds,
    target_bounds: &RasterBounds,
    proj: &impl Projection,
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

            let (tgt_lon, tgt_lat) = proj.inverse_project_unchecked(tgt_x, tgt_y);

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


// DEBUG: Weights and anchors seem fine
pub(crate) fn precompute_internals<F: ResamplingFilter>(
    tgt_ixs_jys: &Array2<IXJYPair>,
    params: &WarperParameters,
) -> Result<Array2<ResamplingKernelInternals>, WarperError> {
    let internals = tgt_ixs_jys.map(|&crds| {
        let anchor_idx = (crds.ix.floor() as u32, crds.jy.floor() as u32);

        let src_x = crds.ix - params.offsets.i as f64;
        let src_y = crds.jy - params.offsets.j as f64;

        let delta_x = src_x - 0.5 - (src_x - 0.5).floor();
        let delta_y = src_y - 0.5 - (src_y - 0.5).floor();

        let x_weights = [-1, 0, 1, 2].map(|i| {
            if params.scales.x < 1.0 {
                F::apply((i as f64 - delta_x) * params.scales.x)
            } else {
                F::apply(i as f64 - delta_x)
            }
        });

        let y_weights = [-1, 0, 1, 2].map(|j| {
            if params.scales.y < 1.0 {
                F::apply((j as f64 - delta_y) * params.scales.y)
            } else {
                F::apply(j as f64 - delta_y)
            }
        });

        ResamplingKernelInternals {
            anchor_idx,
            x_weights,
            y_weights,
        }
    });

    let dbg_anchors_x = internals.map(|intr| {
        intr.anchor_idx.0
    });
    let dbg_anchors_y = internals.map(|intr| {
        intr.anchor_idx.1
    });

    ndarray_npy::write_npy("./test-data/dbg_anchors_x.npy", &dbg_anchors_x).unwrap();
    ndarray_npy::write_npy("./test-data/dbg_anchors_y.npy", &dbg_anchors_y).unwrap();

    Ok(internals)
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;

    use crate::tests::reference_setup;

    use super::precompute_ixs_jys;

    #[test]
    fn ix_jy() {
        let (src_bounds, tgt_bounds, proj) = reference_setup();

        let ixs_jys = precompute_ixs_jys(&src_bounds, &tgt_bounds, &proj).unwrap();

        assert_approx_eq!(f64, ixs_jys[[0, 0]].ix, 4.7102160316373727, epsilon = 1e-6);
        assert_approx_eq!(f64, ixs_jys[[0, 0]].jy, 8.8887293250701873, epsilon = 1e-6);
    }
}
