use mappers::Projection;

use crate::{precompute::{self, precompute_ixs_jys}, GenericXYPair, RasterBounds, RasterBoundsDefinition, ResamplingFilter, SourceXYPair, TargetXYPair, Warper, WarperError, WarperParameters};

impl Warper {
    pub fn warp_gpu<F: ResamplingFilter, SP: Projection, TP: Projection>(
        source_bounds: &RasterBoundsDefinition<SP>,
        target_bounds: &RasterBoundsDefinition<TP>,
    ) -> Result<Self, WarperError> {
        let source_bounds =
            RasterBounds::<SP, GenericXYPair>::from(source_bounds).cast_xy_pairs::<SourceXYPair>();
        let target_bounds =
            RasterBounds::<TP, GenericXYPair>::from(target_bounds).cast_xy_pairs::<TargetXYPair>();

        let params = WarperParameters::compute::<F, SP, TP>(&source_bounds, &target_bounds)?;
        let tgt_ixs_jys = precompute_ixs_jys(&source_bounds, &target_bounds)?;
        let internals = precompute::precompute_internals::<F>(&tgt_ixs_jys, &params);
        let source_shape = (
            source_bounds.shape.j as usize,
            source_bounds.shape.i as usize,
        );

        Ok(Self {
            source_shape,
            internals,
        })
    }
}
