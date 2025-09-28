#![cfg_attr(docsrs, doc(cfg(feature = "io")))]

use std::fs::File;

use ndarray::Array2;
use serde::{Deserialize, Serialize};

#[cfg(feature = "io")]
use crate::helpers::WarperIOError;
use crate::{ResamplingKernelInternals, Warper};

/// Warper uses ndarray which implements unsafe methods.
/// From clippy: Deriving `serde::Deserialize` will create a constructor that may violate invariants held by another constructor.
/// This wrapper prevents deriving `Deserialize` for type with usafe methods.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct WarperCompatIO {
    source_shape: (usize, usize),
    target_shape: (usize, usize),
    internals: Vec<ResamplingKernelInternals>,
}

impl From<Warper> for WarperCompatIO {
    fn from(warper_lib: Warper) -> Self {
        Self {
            source_shape: warper_lib.source_shape,
            target_shape: warper_lib.internals.dim(),
            internals: warper_lib.internals.into_flat().to_vec(),
        }
    }
}

impl TryFrom<WarperCompatIO> for Warper {
    type Error = ndarray::ShapeError;

    fn try_from(warper_io: WarperCompatIO) -> Result<Self, Self::Error> {
        Ok(Self {
            source_shape: warper_io.source_shape,
            internals: Array2::from_shape_vec(warper_io.target_shape, warper_io.internals)?,
        })
    }
}
impl Warper {
    #[cfg_attr(docsrs, doc(cfg(feature = "io")))]
    pub fn save_to_file(self, path: &str) -> Result<(), WarperIOError> {
        let mut file = File::create(path)?;
        let object = WarperCompatIO::from(self);

        bincode::serde::encode_into_std_write(object, &mut file, bincode::config::standard())?;

        Ok(())
    }

    #[cfg_attr(docsrs, doc(cfg(feature = "io")))]
    pub fn load_from_file(path: &str) -> Result<Self, WarperIOError> {
        let mut file = File::open(path)?;

        let warper: WarperCompatIO =
            bincode::serde::decode_from_std_read(&mut file, bincode::config::standard())?;

        Ok(warper.try_into()?)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::fs;

    use anyhow::Result;
    use mappers::projections::{LambertConformalConic, LongitudeLatitude};

    use crate::{filters::CubicBSpline, tests::reference_setup_def, Warper, WarperInitialize};

    #[test]
    fn io() -> Result<()> {
        let (src_bounds, tgt_bounds) = reference_setup_def()?;
        let warper = Warper::initialize::<CubicBSpline, LongitudeLatitude, LambertConformalConic>(
            &src_bounds,
            &tgt_bounds,
        )?;

        warper
            .clone()
            .save_to_file("./tests/data/saved-warper.dat")?;

        let loaded = Warper::load_from_file("./tests/data/saved-warper.dat")?;

        fs::remove_file("./tests/data/saved-warper.dat").unwrap_or(()); // cleanup

        assert_eq!(warper, loaded);

        Ok(())
    }
}
