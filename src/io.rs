use std::fs::{self, File};
use std::path::Path;

use ndarray::Array2;
use rkyv::rancor::Error;
use rkyv::ser::writer::IoWriter;
use rkyv::{Archive, Deserialize, Serialize, access, deserialize};

use crate::helpers::WarperIOError;
use crate::{ResamplingKernelInternals, Warper};

/// Warper uses ndarray which implements unsafe methods.
/// From clippy: Deriving `serde::Deserialize` will create a constructor that may violate invariants held by another constructor.
/// This wrapper prevents deriving `Deserialize` for type with unsafe methods.
/// This issue likely applies also for rkyv
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
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
    pub fn save_to_file<P: AsRef<Path>>(self, path: P) -> Result<(), WarperIOError> {
        let file = File::create(path)?;
        let io_writer = IoWriter::new(file);
        let object = WarperCompatIO::from(self);

        rkyv::api::high::to_bytes_in::<_, Error>(&object, io_writer)?;

        Ok(())
    }

    #[cfg_attr(docsrs, doc(cfg(feature = "io")))]
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, WarperIOError> {
        let file: Vec<u8> = fs::read(path)?;
        let archived = access::<ArchivedWarperCompatIO, Error>(&file)?;
        let warper = deserialize::<WarperCompatIO, Error>(archived)?;

        Ok(warper.try_into()?)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::fs;

    use anyhow::Result;
    use mappers::projections::{LambertConformalConic, LongitudeLatitude};

    use crate::{Warper, filters::CubicBSpline, tests::reference_setup_def};

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
