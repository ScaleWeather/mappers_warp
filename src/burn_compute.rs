use burn::backend::Cuda;
use burn::prelude::*;
use ndarray::{Array2, ArrayView2};

use crate::Warper;

impl Warper {
    #[must_use]
    pub fn warp_gpu_unchecked<'a, A: Into<ArrayView2<'a, f64>>>(
        &self,
        source_raster: A,
    ) -> Array2<f32> {
        let source_raster: ArrayView2<f64> = source_raster.into();

        let device = Default::default();

        let target_shape = self.internals.dim();

        let x_weights = self
            .internals
            .map(|intr| intr.x_weights)
            .flatten()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let x_weights = Tensor::<Cuda, 1>::from_data(x_weights.as_slice(), &device)
            .reshape([target_shape.0, target_shape.1, 1, 4])
            .repeat_dim(2, 4);

        let y_weights = self
            .internals
            .map(|intr| intr.y_weights)
            .flatten()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let y_weights = Tensor::<Cuda, 1>::from_data(y_weights.as_slice(), &device).reshape([
            target_shape.0,
            target_shape.1,
            4,
            1,
        ]);

        let weight_accumulator = x_weights
            .clone()
            .sum_dim(3)
            .mul(y_weights.clone())
            .sum_dim(2);

        let target_raster = self
            .internals
            .map(|intr| {
                source_raster.slice(ndarray::s![
                    (intr.anchor_idx.1 - 1)..(intr.anchor_idx.1 + 3),
                    (intr.anchor_idx.0 - 1)..(intr.anchor_idx.0 + 3)
                ])
            })
            .into_iter()
            .map(|arr| arr.into_iter())
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let target_raster = Tensor::<Cuda, 1>::from_data(target_raster.as_slice(), &device)
            .reshape([target_shape.0, target_shape.1, 4, 4])
            .mul(x_weights)
            .sum_dim(3)
            .mul(y_weights)
            .sum_dim(2)
            .div(weight_accumulator)
            .squeeze_dims::<2>(&[2, 3]);

        let target_raster = target_raster.to_data().into_vec::<f32>().unwrap();
        Array2::from_shape_vec(target_shape, target_raster).unwrap()
    }
}
