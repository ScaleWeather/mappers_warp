use mappers::Projection;
use ndarray::Array2;

use crate::{
    warp_params::WarperParameters, IXJYPair, LonLatPair, RasterBounds, ResamplingFilter,
    ResamplingKernelInternals, WarperError, XYPair,
};

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

        let delta = compute_delta(&crds, params);

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

fn compute_delta(crds: &IXJYPair, params: &WarperParameters) -> XYPair {
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
    use float_cmp::assert_approx_eq;
    use mappers::{projections::LambertConformalConic, Ellipsoid};
    use ndarray::Array2;

    use crate::{
        tests::reference_setup, warp_params::WarperParameters, CubicBSpline, IXJYPair,
        RasterBounds, Warper,
    };

    use super::precompute_ixs_jys;

    #[test]
    fn ix_jy() {
        let (src_bounds, tgt_bounds, proj) = reference_setup();

        let ixs_jys = precompute_ixs_jys(&src_bounds, &tgt_bounds, &proj).unwrap();

        assert_approx_eq!(f64, ixs_jys[[0, 0]].ix, 4.7102160316373727, epsilon = 1e-6);
        assert_approx_eq!(f64, ixs_jys[[0, 0]].jy, 8.8887293250701873, epsilon = 1e-6);
    }

    #[test]
    fn delta() {
        let (src_bounds, tgt_bounds, proj) = reference_setup();

        let params =
            WarperParameters::compute::<CubicBSpline>(&src_bounds, &tgt_bounds, &proj).unwrap();

        let crds = IXJYPair {
            ix: 4.7102160316373727,
            jy: 8.8887293250701873,
        };

        let delta = super::compute_delta(&crds, &params);

        assert_approx_eq!(f64, delta.x, 0.21021603163737268, epsilon = 1e-6);
        assert_approx_eq!(f64, delta.y, 0.38872932507018731, epsilon = 1e-6);
    }

    #[test]
    fn shift_investigation() {
        let source_bounds = RasterBounds::new((60.00, 68.25), (31.75, 40.0), 0.25, 0.25).unwrap();

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

        let warper = Warper::initialize::<LambertConformalConic, CubicBSpline>(
            &source_bounds,
            &target_bounds,
            &proj,
        )
        .unwrap();

        let source_raster: Array2<f64> = ndarray_npy::read_npy("./test-data/waves_34.npy").unwrap();
        let target_raster = warper.warp(&source_raster).unwrap();

        let ixs_jys = precompute_ixs_jys(&source_bounds, &target_bounds, &proj).unwrap();

        let crd = ixs_jys[[38, 28]];

        let params =
            WarperParameters::compute::<CubicBSpline>(&source_bounds, &target_bounds, &proj)
                .unwrap();

        let delta = super::compute_delta(&crd, &params);
        let anchor = warper.internals[[38, 28]].anchor_idx;

        println!("coords: {:?}", crd);
        println!("delta: {:?}", delta);

        println!("value: {:?}", target_raster[[38, 28]]);
        println!("{:?}", warper.internals[[38, 28]]);
        println!(
            "src input 00: {:?}",
            source_raster[[anchor.1 as usize, anchor.0 as usize]]
        );
    }
}
