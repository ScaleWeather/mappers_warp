use mappers::Projection;
use ndarray::Array2;

use crate::{IXJYPair, LonLatPair, RasterBounds, WarperError, XYPair};

pub(crate) fn precompute_ixs_jys(
    source_bounds: &RasterBounds,
    target_bounds: &RasterBounds,
    proj: &impl Projection,
) -> Result<Array2<IXJYPair>, WarperError> {
    let tgt_ur_edge_corner = XYPair {
        x: target_bounds.min.x - (0.5 * target_bounds.spacing.x),
        y: target_bounds.max.y + (0.5 * target_bounds.spacing.y),
    };
    let src_ur_edge_corner = LonLatPair {
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
        |(i, j)| {
            // 0.5 shift is because we are measuring from edge corner to midpoint
            let tgt_x = tgt_ur_edge_corner.x + ((i as f64 + 0.5) * target_bounds.spacing.x);
            let tgt_y = tgt_ur_edge_corner.y - ((j as f64 + 0.5) * target_bounds.spacing.y);

            let (tgt_lon, tgt_lat) = proj.inverse_project_unchecked(tgt_x, tgt_y);

            IXJYPair {
                ix: (tgt_lon - src_ur_edge_corner.lon) * conversion_scaling.x,
                jy: (src_ur_edge_corner.lat - tgt_lat) * conversion_scaling.y,
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
