use anyhow::{Context, Result};
use ndarray::Array2;

pub fn open_nc_data(filename: &str) -> Result<Array2<f64>> {
    let file = netcdf::open(filename)?;
    let var = file.variable("data_arr").context("")?;

    let data = var.get::<f64, _>(..)?;
    let data = data.into_dimensionality()?;

    Ok(data)
}

#[allow(unused)]
pub trait NdStats {
    fn min(&self) -> Result<f64>;
    fn max(&self) -> Result<f64>;
}

impl NdStats for Array2<f64> {
    fn max(&self) -> Result<f64> {
        let res = self.iter().max_by(|x, y| x.total_cmp(y)).context("")?;
        Ok(*res)
    }

    fn min(&self) -> Result<f64> {
        let res = self.iter().min_by(|x, y| x.total_cmp(y)).context("")?;
        Ok(*res)
    }
}
