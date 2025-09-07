#![allow(unused)]

use cubecl::frontend::Floor;
use cubecl::prelude::*;
use mappers::Projection;
use ndarray::{Array2, ArrayView2};

use crate::{
    GenericXYPair, IJPair, IXJYPair, RasterBounds, RasterBoundsDefinition, ResamplingFilter,
    SourceXYPair, TargetXYPair, Warper, WarperParameters, filters_gpu::ResamplingFilterGPU,
    precompute::precompute_ixs_jys,
};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, CubeLaunch, CubeType)]
pub(super) struct WarperParametersGPU {
    pub scales: GenericXYPair,
    pub offsets: IJPair,
}

impl Warper {
    pub fn warp_unchecked_gpu<
        'a,
        F: ResamplingFilter,
        SP: Projection,
        TP: Projection,
        A: Into<ArrayView2<'a, f64>>,
    >(
        source_bounds: &RasterBoundsDefinition<SP>,
        target_bounds: &RasterBoundsDefinition<TP>,
        source_raster: A,
    ) -> Array2<f64> {
        let source_raster: ArrayView2<f64> = source_raster.into();

        let source_bounds =
            RasterBounds::<SP, GenericXYPair>::from(source_bounds).cast_xy_pairs::<SourceXYPair>();
        let target_bounds =
            RasterBounds::<TP, GenericXYPair>::from(target_bounds).cast_xy_pairs::<TargetXYPair>();

        let params =
            WarperParameters::compute::<F, SP, TP>(&source_bounds, &target_bounds).unwrap();
        let tgt_ixs_jys = precompute_ixs_jys(&source_bounds, &target_bounds).unwrap();

        todo!()
    }
}

#[cube(launch)]
fn precompute_internals<F: ResamplingFilterGPU>(
    tgt_ixs: &Tensor<f64>,
    tgt_jys: &Tensor<f64>,
    source_raster: &Tensor<f64>,
    result: &mut Tensor<f64>,
    params: &WarperParametersGPU,
) {
    if ABSOLUTE_POS_X >= result.shape(1) || ABSOLUTE_POS_Y >= result.shape(0) {
        terminate!()
    }

    let target_raster_index = (ABSOLUTE_POS_Y * result.stride(1)) + ABSOLUTE_POS_X;

    let crds = IXJYPair {
        ix: tgt_ixs[target_raster_index],
        jy: tgt_jys[target_raster_index],
    };

    let anchor_idx: (f64, f64) = (Floor::floor(crds.ix - 0.5), Floor::floor(crds.jy - 0.5));
    let anchor_idx = (anchor_idx.0 as u32, anchor_idx.1 as u32);

    let delta = {
        let src_x = crds.ix - (params.offsets.i as f64);
        let src_y = crds.jy - (params.offsets.j as f64);

        GenericXYPair {
            x: src_x - 0.5 - Floor::floor(src_x - 0.5),
            y: src_y - 0.5 - Floor::floor(src_y - 0.5),
        }
    };

    // for now I can't think of a more concise solution
    let mut x_weights = Array::<f64>::new(4);
    let mut y_weights = Array::<f64>::new(4);

    #[unroll]
    for i in 0..4_u32 {
        let offset = (i - 1) as f64;

        if params.scales.x < 1.0 {
            x_weights[i] = F::apply((offset - delta.x) * params.scales.x);
        } else {
            x_weights[i] = F::apply(offset - delta.x);
        }

        if params.scales.y < 1.0 {
            y_weights[i] = F::apply((offset - delta.y) * params.scales.y);
        } else {
            y_weights[i] = F::apply(offset - delta.y);
        }
    }

    let mut weight_accum = <f64 as Float>::new(0.0);
    let mut result_accum = <f64 as Float>::new(0.0);

    for j in 0..4 {
        let mut inner_weight_accum = <f64 as Float>::new(0.0);
        let mut inner_result_accum = <f64 as Float>::new(0.0);

        for i in 0..4 {
            let value = {
                let x = i - 1;
                let y = j - 1;
                let idx = (y * source_raster.stride(1)) + x;
                source_raster[idx]
            };
            let x_weight = x_weights[i];

            inner_weight_accum += x_weight;
            inner_result_accum += x_weight * value;
        }

        let y_weight = y_weights[j];

        weight_accum += inner_weight_accum * y_weight;
        result_accum += inner_result_accum * y_weight;
    }

    result[target_raster_index] = result_accum / weight_accum;
}
