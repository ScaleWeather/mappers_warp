use mappers::Projection;
use ndarray::{concatenate, stack, Array, Array2, Axis};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum WarperError {
    #[error("Invalid raster dimensions.")]
    InvalidRasterDimensions,

    #[error("Ndarray error.")]
    NdarrayError(#[from] ndarray::ShapeError),

    #[error("Projection error.")]
    ProjectionError(#[from] mappers::ProjectionError),
}

#[cfg(feature = "io")]
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WarperIOError {
    #[error("File not found.")]
    FileNotFound,
    #[error("Invalid file.")]
    InvalidFile,
}

pub trait ResamplingFilter {
    fn apply(x: f64) -> f64;

    const X_RADIUS: f64;
    const Y_RADIUS: f64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CubicBSpline;

impl ResamplingFilter for CubicBSpline {
    fn apply(x: f64) -> f64 {
        todo!()
    }

    const X_RADIUS: f64 = 2.0;
    const Y_RADIUS: f64 = 2.0;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MitchellNetravali;

impl ResamplingFilter for MitchellNetravali {
    fn apply(x: f64) -> f64 {
        todo!()
    }

    const X_RADIUS: f64 = 2.0;
    const Y_RADIUS: f64 = 2.0;
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct XYPair {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct LonLatPair {
    pub lon: f64,
    pub lat: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct IXJYPair {
    pub ix: f64,
    pub jy: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct IJPair {
    pub i: u32,
    pub j: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct RasterBounds {
    pub min: XYPair,
    pub max: XYPair,
    pub spacing: XYPair,
    pub shape: IJPair,
}

impl RasterBounds {
    pub fn new(
        x_bounds: (f64, f64),
        y_bounds: (f64, f64),
        dx: f64,
        dy: f64,
    ) -> Result<Self, WarperError> {
        let (min_x, max_x) = x_bounds;
        let (min_y, max_y) = y_bounds;

        if min_x >= max_x || min_y >= max_y {
            return Err(WarperError::InvalidRasterDimensions);
        }

        let nx = (max_x - min_x) / dx;
        let ny = (max_y - min_y) / dy;

        if nx.fract() != 0.0 || ny.fract() != 0.0 {
            return Err(WarperError::InvalidRasterDimensions);
        }

        let nx = nx as u32 + 1;
        let ny = ny as u32 + 1;

        Ok(Self {
            min: XYPair { x: min_x, y: min_y },
            max: XYPair { x: max_x, y: max_y },
            spacing: XYPair { x: dx, y: dy },
            shape: IJPair { i: nx, j: ny },
        })
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
struct ResamplingKernelInternals {
    anchor_pixel_idx: [u32; 2],
    x_weights: [f64; 4],
    y_weights: [f64; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WarperBuilder {}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Warper {
    internals: ResamplingKernelInternals,
}

impl Warper {
    pub fn initialize(
        source_bounds: &RasterBounds,
        target_bounds: &RasterBounds,
        proj: &impl Projection,
        kernel: &impl ResamplingFilter,
    ) -> Result<Self, WarperError> {
        let (tgt_min_extrema, tgt_max_extrema) =
            compute_target_outer_extrema(source_bounds, target_bounds, proj)?;

        todo!()
    }

    pub fn warp(&self, lonlat_raster: &Array2<f64>) -> Result<Array2<f64>, WarperError> {
        todo!()
    }

    #[cfg(feature = "io")]
    pub fn save_to_file(&self, path: &str) -> Result<(), WarperIOError> {
        todo!()
    }

    #[cfg(feature = "io")]
    pub fn load_from_file(path: &str) -> Result<Self, WarperIOError> {
        todo!()
    }
}

fn compute_target_outer_extrema(
    source_bounds: &RasterBounds,
    target_bounds: &RasterBounds,
    proj: &impl Projection,
) -> Result<(IXJYPair, IXJYPair), WarperError> {
    let (min_extr, max_extr) = get_target_extrema_lonlat(target_bounds, proj)?;

    // Shift here is because extrema are computed at edges
    let min_x_out = ((min_extr.lon - source_bounds.min.x) / source_bounds.spacing.x) + 0.5;
    let max_x_out = ((max_extr.lon - source_bounds.min.x) / source_bounds.spacing.x) + 0.5;
    let max_y_out = ((source_bounds.max.y - min_extr.lat) / source_bounds.spacing.y) + 0.5;
    let min_y_out = ((source_bounds.max.y - max_extr.lat) / source_bounds.spacing.y) + 0.5;

    Ok((
        IXJYPair {
            ix: min_x_out,
            jy: min_y_out,
        },
        IXJYPair {
            ix: max_x_out,
            jy: max_y_out,
        },
    ))
}

fn get_target_extrema_lonlat(
    target_bounds: &RasterBounds,
    proj: &impl Projection,
) -> Result<(LonLatPair, LonLatPair), WarperError> {
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

    let mut min_lon = f64::INFINITY;
    let mut max_lon = f64::NEG_INFINITY;

    let mut min_lat = f64::INFINITY;
    let mut max_lat = f64::NEG_INFINITY;

    edges_xy
        .rows()
        .into_iter()
        .try_for_each(|xy| -> Result<(), WarperError> {
            let (lon, lat) = proj.inverse_project(xy[0], xy[1])?;

            min_lon = min_lon.min(lon);
            max_lon = max_lon.max(lon);

            min_lat = min_lat.min(lat);
            max_lat = max_lat.max(lat);

            Ok(())
        })?;

    Ok((
        LonLatPair {
            lon: min_lon,
            lat: min_lat,
        },
        LonLatPair {
            lon: max_lon,
            lat: max_lat,
        },
    ))
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;
    use mappers::{projections::LambertConformalConic, Ellipsoid};

    use crate::{compute_target_outer_extrema, CubicBSpline, RasterBounds};

    fn reference_setup() -> (
        RasterBounds,
        RasterBounds,
        LambertConformalConic,
        CubicBSpline,
    ) {
        let source_bounds = RasterBounds::new((60.00, 67.25), (32.75, 40.0), 0.25, 0.25).unwrap();

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

        let kernel = CubicBSpline;

        return (source_bounds, target_bounds, proj, kernel);
    }

    #[test]
    fn unprojected_extrema() {
        let (source_bounds, target_bounds, proj, _) = reference_setup();

        let (min_extrema, max_extrema) =
            compute_target_outer_extrema(&source_bounds, &target_bounds, &proj).unwrap();

        assert_approx_eq!(f64, min_extrema.ix, 4.457122955747991, epsilon = 1e-6);
        assert_approx_eq!(f64, min_extrema.jy, 6.9363298550977959, epsilon = 1e-6);
        assert_approx_eq!(f64, max_extrema.ix, 26.145584743939651, epsilon = 1e-6);
        assert_approx_eq!(f64, max_extrema.jy, 28.72260733112293, epsilon = 1e-6);
    }
}
