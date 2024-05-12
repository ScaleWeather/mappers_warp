pub trait ResamplingFilter {
    fn apply(x: f64) -> f64;

    const X_RADIUS: f64;
    const Y_RADIUS: f64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CubicBSpline;

impl ResamplingFilter for CubicBSpline {
    fn apply(x: f64) -> f64 {
        let xp2 = x + 2.0;
        let xp1 = x + 1.0;
        let xm1 = x - 1.0;

        let xp2c = xp2 * xp2 * xp2;

        let mut res = 0.0;

        if xm1 > 0.0 {
            res += -4.0 * xm1 * xm1 * xm1;
        };

        if x > 0.0 {
            res += 6.0 * x * x * x;
        };

        if xp1 > 0.0 {
            res += -4.0 * xp1 * xp1 * xp1;
        };

        if xp2 > 0.0 {
            res += xp2c;
        };

        res
    }

    const X_RADIUS: f64 = 2.0;
    const Y_RADIUS: f64 = 2.0;
}

/// B = 1/3, C = 1/3
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MitchellNetravali;

impl ResamplingFilter for MitchellNetravali {
    const X_RADIUS: f64 = 2.0;
    const Y_RADIUS: f64 = 2.0;

    fn apply(x: f64) -> f64 {
        let n = x.abs();

        if n < 1.0 {
            (7.0 / 6.0) * n.powi(3) - 2.0 * n.powi(2) + 8.0 / 9.0
        } else if (1.0..2.0).contains(&n) {
            -7.0 / 18.0 * n.powi(3) + 2.0 * n.powi(2) - 10.0 / 3.0 * n + 16.0 / 9.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;

    use crate::{CubicBSpline, ResamplingFilter};

    #[test]
    fn bspline_filter() {
        assert_approx_eq!(f64, CubicBSpline::apply(1.675), 0.034_328_1, epsilon = 1e-5);
        assert_approx_eq!(f64, CubicBSpline::apply(1.231), 0.454_757, epsilon = 1e-5);
        assert_approx_eq!(f64, CubicBSpline::apply(0.115), 3.92521, epsilon = 1e-5);
        assert_approx_eq!(f64, CubicBSpline::apply(-0.243), 3.68875, epsilon = 1e-5);
        assert_approx_eq!(f64, CubicBSpline::apply(-1.65), 0.042_875, epsilon = 1e-5);
    }
}
