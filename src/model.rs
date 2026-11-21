use gpv_products::{
    model::{BandSpec, LngLatGrid},
    products::GpvProductIdentifier,
};

use crate::{
    quantize::{BandOmit, BandQuantize},
    tile::Reducer,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct CompactOptI32(i32);

impl CompactOptI32 {
    pub const NONE: Self = Self(i32::MIN);

    #[inline]
    pub fn new(value: Option<i32>) -> Self {
        match value {
            Some(i32::MIN) => panic!("i32::MIN is reserved for missing values"),
            Some(value) => Self(value),
            None => Self::NONE,
        }
    }

    #[inline]
    pub fn get(self) -> Option<i32> {
        (self != Self::NONE).then_some(self.0)
    }

    #[inline]
    pub fn unwrap_or(self, default: i32) -> i32 {
        self.get().unwrap_or(default)
    }

    #[inline]
    pub fn map(self, f: impl FnOnce(i32) -> Self) -> Self {
        self.get().map(f).unwrap_or(Self::NONE)
    }
}

impl PartialOrd for CompactOptI32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CompactOptI32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.get().cmp(&other.get())
    }
}

impl IntoIterator for CompactOptI32 {
    type Item = i32;
    type IntoIter = std::option::IntoIter<i32>;

    fn into_iter(self) -> Self::IntoIter {
        self.get().into_iter()
    }
}

impl IntoIterator for &CompactOptI32 {
    type Item = i32;
    type IntoIter = std::option::IntoIter<i32>;

    fn into_iter(self) -> Self::IntoIter {
        self.get().into_iter()
    }
}

#[derive(Debug)]
pub(crate) struct Band {
    pub values: Vec<CompactOptI32>,
}

#[derive(Debug)]
pub(crate) struct BaseTile {
    pub point_positions: Vec<(u16, u16)>,
    pub point_powers: Vec<u8>,
    /// Where each point's own chunk lies, relative to this one.
    ///
    /// A cell is copied into every chunk its margin reaches, so a tile reading
    /// several chunks sees a cell near their shared edge more than once. A
    /// maximum does not care, but a weighted statistic would count the cell
    /// twice. The chunk holding the cell's centre is its own; this records the
    /// offset to it, so that exactly one copy can be kept.
    pub point_homes: Vec<(i8, i8)>,
    pub bands: Vec<Band>,
}

#[derive(Debug)]
pub(crate) struct TilesetSpec {
    pub name: String,
    pub base_z: u8,
    pub grid_spec: LngLatGrid,
    /// How the cells of a block are combined; the product's own aggregation
    /// unless one was chosen.
    pub reducer: Reducer,
    /// Raw value counted over the part of a block no cell is stored for, per
    /// band; `None` leaves that part out.
    pub fill: Vec<Option<i32>>,
    pub band_specs: Vec<BandSpec>,
    /// Optional quantization per band, in `band_specs` order.
    pub quantize: Vec<Option<BandQuantize>>,
    /// Physical zero dropped before quantization, in `band_specs` order.
    pub omit_zero: Vec<Option<BandOmit>>,
    /// Quantized output classes dropped from the tile, in `band_specs` order.
    pub omit_class: Vec<Option<BandOmit>>,
    pub bounds: [f64; 4],
}

#[derive(Debug)]
pub(crate) struct PreparedProduct {
    pub product_id: GpvProductIdentifier,
    pub spec: TilesetSpec,
    pub chunks: Vec<(u64, BaseTile)>,
}

impl PreparedProduct {
    pub fn has_chunks_in_range(&self, begin: u64, end: u64) -> bool {
        let begin_index = self.chunks.partition_point(|(id, _)| *id < begin);
        self.chunks
            .get(begin_index)
            .is_some_and(|(id, _)| *id < end)
    }

    pub fn chunks_in_range(&self, begin: u64, end: u64) -> &[(u64, BaseTile)] {
        let begin_index = self.chunks.partition_point(|(id, _)| *id < begin);
        let end_index =
            begin_index + self.chunks[begin_index..].partition_point(|(id, _)| *id < end);
        &self.chunks[begin_index..end_index]
    }
}
