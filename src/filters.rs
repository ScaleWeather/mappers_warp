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
        } else if n >= 1.0 && n < 2.0 {
            -7.0 / 18.0 * n.powi(3) + 2.0 * n.powi(2) - 10.0 / 3.0 * n + 16.0 / 9.0
        } else {
            0.0
        }
    }
}
