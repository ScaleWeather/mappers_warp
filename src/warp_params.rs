use mappers::{ConversionPipe, Projection};
use ndarray::{concatenate, stack, Array, Axis};

use crate::{
    GenericXYPair, IJPair, IXJYPair, MinMaxPair, RasterBounds, ResamplingFilter, SourceXYPair,
    TargetXYPair, WarperError,
};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub(super) struct WarperParameters {
    pub scales: GenericXYPair,
    pub offsets: IJPair,
}

impl WarperParameters {
    pub fn compute<F: ResamplingFilter, SP: Projection, TP: Projection>(
        source_bounds: &RasterBounds<SP, SourceXYPair>,
        target_bounds: &RasterBounds<TP, TargetXYPair>,
    ) -> Result<Self, WarperError> {
        let wrap_margin = F::X_RADIUS.max(F::Y_RADIUS) as u32;

        let tgt_extrema = compute_target_outer_extrema(source_bounds, target_bounds)?;

        let clamped_extrema =
            compute_clamped_extrema(&tgt_extrema, source_bounds.shape, wrap_margin)?;

        let (offsets, scales) = compute_offsets_and_scales(
            &tgt_extrema,
            &clamped_extrema,
            source_bounds,
            target_bounds,
            IJPair {
                i: F::X_RADIUS as u32,
                j: F::Y_RADIUS as u32,
            },
        );

        Ok(WarperParameters { scales, offsets })
    }
}

fn compute_target_outer_extrema<SP: Projection, TP: Projection>(
    source_bounds: &RasterBounds<SP, SourceXYPair>,
    target_bounds: &RasterBounds<TP, TargetXYPair>,
) -> Result<MinMaxPair<IXJYPair>, WarperError> {
    let proj_pipe = &target_bounds.proj.pipe_to(&source_bounds.proj);
    let tgt_extr = get_target_extrema_on_source(target_bounds, proj_pipe)?;

    // Shift here is because extrema are computed at edges
    let min_x_out = ((tgt_extr.min.x - source_bounds.min.x) / source_bounds.spacing.x) + 0.5;
    let max_x_out = ((tgt_extr.max.x - source_bounds.min.x) / source_bounds.spacing.x) + 0.5;
    let max_y_out = ((source_bounds.max.y - tgt_extr.min.y) / source_bounds.spacing.y) + 0.5;
    let min_y_out = ((source_bounds.max.y - tgt_extr.max.y) / source_bounds.spacing.y) + 0.5;

    Ok(MinMaxPair {
        min: IXJYPair {
            ix: min_x_out,
            jy: min_y_out,
        },
        max: IXJYPair {
            ix: max_x_out,
            jy: max_y_out,
        },
    })
}

fn compute_clamped_extrema(
    tgt_extr: &MinMaxPair<IXJYPair>,
    src_shape: IJPair,
    min_margin: u32,
) -> Result<MinMaxPair<IJPair>, WarperError> {
    if tgt_extr.min.ix < f64::from(min_margin)
        || tgt_extr.min.jy < f64::from(min_margin)
        || tgt_extr.max.ix > f64::from(src_shape.i - min_margin)
        || tgt_extr.max.jy > f64::from(src_shape.j - min_margin)
    {
        return Err(WarperError::SourceRasterTooSmall);
    }

    let n_min_x_out_clamped = tgt_extr.min.ix.floor() as u32;
    let n_min_y_out_clamped = tgt_extr.min.jy.floor() as u32;

    let n_max_x_out_clamped = tgt_extr.max.ix.ceil() as u32;
    let n_max_y_out_clamped = tgt_extr.max.jy.ceil() as u32;

    Ok(MinMaxPair {
        min: IJPair {
            i: n_min_x_out_clamped,
            j: n_min_y_out_clamped,
        },
        max: IJPair {
            i: n_max_x_out_clamped,
            j: n_max_y_out_clamped,
        },
    })
}

fn compute_offsets_and_scales<SP: Projection, TP: Projection>(
    tgt_extrema: &MinMaxPair<IXJYPair>,
    clamped_extrema: &MinMaxPair<IJPair>,
    source_bounds: &RasterBounds<SP, SourceXYPair>,
    target_bounds: &RasterBounds<TP, TargetXYPair>,
    kernel_radius: IJPair,
) -> (IJPair, GenericXYPair) {
    let offsets = compute_src_offsets(clamped_extrema.min, source_bounds.shape, kernel_radius);

    let src_x_size_raw = f64::from(source_bounds.shape.i - clamped_extrema.min.i)
        .min(tgt_extrema.max.ix - tgt_extrema.min.ix)
        .max(0.0);
    let src_y_size_raw = f64::from(source_bounds.shape.j - clamped_extrema.min.j)
        .min(tgt_extrema.max.jy - tgt_extrema.min.jy)
        .max(0.0);

    let src_x_size = (source_bounds.shape.i as i32 - offsets.i as i32)
        .min(clamped_extrema.max.i as i32 - offsets.i as i32 + kernel_radius.i as i32)
        .max(0) as u32;
    let src_y_size = (source_bounds.shape.j as i32 - offsets.j as i32)
        .min(clamped_extrema.max.j as i32 - offsets.j as i32 + kernel_radius.j as i32)
        .max(0) as u32;

    let src_x_extra_size = f64::from(src_x_size) - src_x_size_raw;
    let src_y_extra_size = f64::from(src_y_size) - src_y_size_raw;

    let x_scale = f64::from(target_bounds.shape.i) / (f64::from(src_x_size) - src_x_extra_size);
    let y_scale = f64::from(target_bounds.shape.j) / (f64::from(src_y_size) - src_y_extra_size);

    (
        IJPair {
            i: offsets.i,
            j: offsets.j,
        },
        GenericXYPair {
            x: x_scale,
            y: y_scale,
        },
    )
}

fn compute_src_offsets(clamped_min: IJPair, src_shape: IJPair, kernel_radius: IJPair) -> IJPair {
    let n_src_x_off = clamped_min
        .i
        .saturating_sub(kernel_radius.i)
        .min(src_shape.i)
        .max(0);
    let n_src_y_off = clamped_min
        .j
        .saturating_sub(kernel_radius.j)
        .min(src_shape.j)
        .max(0);

    IJPair {
        i: n_src_x_off,
        j: n_src_y_off,
    }
}

fn get_target_extrema_on_source<SP: Projection, TP: Projection>(
    target_bounds: &RasterBounds<TP, TargetXYPair>,
    proj_pipe: &ConversionPipe<TP, SP>,
) -> Result<MinMaxPair<SourceXYPair>, WarperError> {
    let x_min = target_bounds.min.x - (0.5 * target_bounds.spacing.x);
    let x_max = target_bounds.max.x + (0.5 * target_bounds.spacing.x);
    let y_min = target_bounds.min.y - (0.5 * target_bounds.spacing.y);
    let y_max = target_bounds.max.y + (0.5 * target_bounds.spacing.y);

    let u_edge_x = Array::range(x_min, x_max, target_bounds.spacing.x);
    let u_edge_y = Array::from_elem(u_edge_x.raw_dim(), y_max);

    let r_edge_y = Array::range(y_max, y_min, -target_bounds.spacing.y);
    let r_edge_x = Array::from_elem(r_edge_y.raw_dim(), x_max);

    let b_edge_x = Array::range(x_max, x_min, -target_bounds.spacing.x);
    let b_edge_y = Array::from_elem(b_edge_x.raw_dim(), y_min);

    let l_edge_y = Array::range(y_min, y_max, target_bounds.spacing.y);
    let l_edge_x = Array::from_elem(l_edge_y.raw_dim(), x_min);

    let u_edge_xy = stack(Axis(1), &[u_edge_x.view(), u_edge_y.view()])?;
    let r_edge_xy = stack(Axis(1), &[r_edge_x.view(), r_edge_y.view()])?;
    let b_edge_xy = stack(Axis(1), &[b_edge_x.view(), b_edge_y.view()])?;
    let l_edge_xy = stack(Axis(1), &[l_edge_x.view(), l_edge_y.view()])?;

    let edges_xy = concatenate(
        Axis(0),
        &[
            u_edge_xy.view(),
            r_edge_xy.view(),
            b_edge_xy.view(),
            l_edge_xy.view(),
        ],
    )?;

    let mut min_src_x = f64::INFINITY;
    let mut max_src_x = f64::NEG_INFINITY;

    let mut min_src_y = f64::INFINITY;
    let mut max_src_y = f64::NEG_INFINITY;

    edges_xy
        .rows()
        .into_iter()
        .try_for_each(|xy| -> Result<(), WarperError> {
            let (src_x, src_y) = proj_pipe.convert(xy[0], xy[1])?;

            min_src_x = min_src_x.min(src_x);
            max_src_x = max_src_x.max(src_x);

            min_src_y = min_src_y.min(src_y);
            max_src_y = max_src_y.max(src_y);

            Ok(())
        })?;

    Ok(MinMaxPair {
        min: SourceXYPair {
            x: min_src_x,
            y: min_src_y,
        },
        max: SourceXYPair {
            x: max_src_x,
            y: max_src_y,
        },
    })
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use float_cmp::assert_approx_eq;

    use super::{compute_clamped_extrema, compute_target_outer_extrema};
    use crate::{
        tests::reference_setup, warp_params::compute_offsets_and_scales, CubicBSpline, IJPair,
        ResamplingFilter, SourceXYPair, TargetXYPair,
    };

    #[test]
    fn assert_with_sample_values() -> Result<()> {
        let (source_bounds, target_bounds) = reference_setup()?;

        let source_bounds = source_bounds.cast_xy_pairs::<SourceXYPair>();
        let target_bounds = target_bounds.cast_xy_pairs::<TargetXYPair>();

        let extrema = compute_target_outer_extrema(&source_bounds, &target_bounds).unwrap();

        assert_approx_eq!(f64, extrema.min.ix, 4.457_122_955_747_991, epsilon = 1e-6);
        assert_approx_eq!(f64, extrema.min.jy, 6.936_329_855_097_795_9, epsilon = 1e-6);
        assert_approx_eq!(f64, extrema.max.ix, 26.145_584_743_939_651, epsilon = 1e-6);
        assert_approx_eq!(f64, extrema.max.jy, 28.722_607_331_122_93, epsilon = 1e-6);

        let clamped_extrema = compute_clamped_extrema(&extrema, source_bounds.shape, 1)?;

        assert_eq!(clamped_extrema.min.i, 4);
        assert_eq!(clamped_extrema.min.j, 6);
        assert_eq!(clamped_extrema.max.i, 27);
        assert_eq!(clamped_extrema.max.j, 29);

        let (offsets, scales) = compute_offsets_and_scales(
            &extrema,
            &clamped_extrema,
            &source_bounds,
            &target_bounds,
            IJPair {
                i: CubicBSpline::X_RADIUS as u32,
                j: CubicBSpline::Y_RADIUS as u32,
            },
        );

        assert_eq!(offsets.i, 2);
        assert_eq!(offsets.j, 4);
        assert_approx_eq!(f64, scales.x, 1.982_621_009_269_152_7, epsilon = 1e-6);
        assert_approx_eq!(f64, scales.y, 2.570_425_354_291_278_3, epsilon = 1e-6);

        Ok(())
    }
}
