use cubecl::prelude::*;

use crate::CubicBSpline;

#[cube]
pub trait ResamplingFilterGPU: Send + Sync + 'static {
    fn apply(x: f64) -> f64;
}

#[cube]
impl ResamplingFilterGPU for CubicBSpline {
    fn apply(x: f64) -> f64 {
        let xp2 = x + 2.0;
        let xp1 = x + 1.0;
        let xm1 = x - 1.0;

        let xp2c = xp2 * xp2 * xp2;

        let mut res = 0.0;

        if xm1 > 0.0 {
            res += -4.0 * xm1 * xm1 * xm1;
        }

        if x > 0.0 {
            res += 6.0 * x * x * x;
        }

        if xp1 > 0.0 {
            res += -4.0 * xp1 * xp1 * xp1;
        }

        if xp2 > 0.0 {
            res += xp2c;
        }

        res
    }
}
