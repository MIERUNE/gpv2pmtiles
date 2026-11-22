//! A pyramid of pixel statistics over the global pixel lattices, built once
//! per product.
//!
//! A zoom that aggregates into pixels gathers, at level `zoom + detail`, the
//! basic cells whose centres fall in each pixel; see
//! [`super::raster::aggregate_pixels`]. The pixel edges of one level are among
//! those of the next, so a pixel's cells are exactly the cells of the four
//! pixels under it. A maximum, a minimum, the bits or a weighted sum of the
//! four, with their weights added, is that of the cells ([`Partial`]).
//! Gathering the cells once, at the finest level any zoom asks for, and
//! halving from there gives every coarser level without reading a cell again.
//!
//! A percentile cannot be recovered from the percentiles below it, and still
//! gathers from the cells.
//!
//! A pixel that holds no cell centre - where a cell is wider or taller than a
//! pixel - takes the cell under its own centre instead, which does not nest.
//! A tile with such a pixel is gathered from the cells as before; see
//! [`super::raster::nested_pyramid`].
//!
//! A level is kept in square blocks of pixels, allocated only where a cell
//! falls. Rain covers a few percent of its grid, so a dense array over the grid
//! was mostly empty, while a sorted list of single pixels spent its time
//! sorting; a block is found by a lookup and filled in place.

//!
//! The partial statistics live only while the next level is made from them;
//! each level keeps the finished value of its pixels.

use foldhash::HashMap;
use rayon::prelude::*;

use gpv_products::model::LngLatGrid as GridSpec;

use super::{Reducer, reduce::Partial};
use crate::{geo::web_mercator_to_lnglat, model::PreparedProduct};

/// Pixels across a block, as a power of two.
const BLOCK_SHIFT: u32 = 6;
const BLOCK: usize = 1 << BLOCK_SHIFT;
const BLOCK_MASK: i64 = BLOCK as i64 - 1;

/// Blocks of `BLOCK` × `BLOCK` pixels, each holding `bands` values per pixel
/// row by row, found by their global block row and column.
struct Blocks<T> {
    index: HashMap<(i64, i64), usize>,
    blocks: Vec<((i64, i64), Vec<T>)>,
}

impl<T> Default for Blocks<T> {
    fn default() -> Self {
        Self {
            index: HashMap::default(),
            blocks: Vec::new(),
        }
    }
}

impl<T> Blocks<T> {
    fn get(&self, key: (i64, i64)) -> Option<&[T]> {
        self.index.get(&key).map(|&at| self.blocks[at].1.as_slice())
    }

    fn from_blocks(blocks: Vec<((i64, i64), Vec<T>)>) -> Self {
        let index = blocks
            .iter()
            .enumerate()
            .map(|(at, (key, _))| (*key, at))
            .collect();
        Self { index, blocks }
    }
}

pub(crate) struct Pyramid {
    /// Levels from `first` up; `levels[i]` is level `first + i`. A pixel
    /// holds its finished value per band, NaN where it has none.
    first: u8,
    levels: Vec<Blocks<f64>>,
    bands: usize,
}

impl std::fmt::Debug for Pyramid {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Pyramid")
            .field("first", &self.first)
            .field(
                "blocks",
                &self
                    .levels
                    .iter()
                    .map(|level| level.blocks.len())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Pyramid {
    /// Whether a pyramid can hold the statistic.
    pub(crate) fn supports(reducer: Reducer) -> bool {
        Partial::supports(reducer)
    }

    /// Gathers the product's cells into level `finest` and halves down to
    /// `coarsest`.
    ///
    /// With `every_band`, a cell counts only where every band has a value, as
    /// the two components of a vector field must.
    pub(crate) fn build(
        product: &PreparedProduct,
        coarsest: u8,
        finest: u8,
        every_band: bool,
    ) -> Self {
        let spec = &product.spec;
        let bands = spec.band_specs.len();
        let reducer = spec.reducer;
        let (max_x, max_y) = grid_extent(product);
        let mut partials = gather(product, finest, (max_x, max_y), every_band);
        let mut levels = Vec::new();
        for level in (coarsest..=finest).rev() {
            let columns = Axis::columns(&spec.grid_spec, level, max_x);
            let rows = Axis::rows(&spec.grid_spec, level, max_y);
            levels.push(Blocks::from_blocks(
                partials
                    .blocks
                    .par_iter()
                    .map(|&(key, ref block)| {
                        let mut out = vec![f64::NAN; block.len()];
                        for row in 0..BLOCK {
                            let height = rows.span(key.0 * BLOCK as i64 + row as i64);
                            for column in 0..BLOCK {
                                let width = columns.span(key.1 * BLOCK as i64 + column as i64);
                                let at = (row * BLOCK + column) * bands;
                                for band in 0..bands {
                                    if let Some(value) = block[at + band].finish(
                                        reducer,
                                        height * width,
                                        spec.fill[band],
                                    ) {
                                        out[at + band] = value;
                                    }
                                }
                            }
                        }
                        (key, out)
                    })
                    .collect(),
            ));
            if level > coarsest {
                partials = halve(&partials, bands, reducer);
            }
        }
        levels.reverse();
        Self {
            first: coarsest,
            levels,
            bands,
        }
    }

    /// Whether the pyramid holds `level`.
    pub(crate) fn has_level(&self, level: u8) -> bool {
        (self.first..self.first + self.levels.len() as u8).contains(&level)
    }

    /// The pixels of a square window of `level` that hold a value, as
    /// `(row, column, values)` relative to the window's corner, row by row,
    /// with NaN for a band without one. Columns wrap round the world.
    pub(crate) fn window(
        &self,
        level: u8,
        (top, left): (i64, i64),
        side: usize,
        mut visit: impl FnMut(usize, usize, &[f64]),
    ) {
        let blocks = &self.levels[usize::from(level - self.first)];
        let world = 1i64 << level;
        let bands = self.bands;
        for row in 0..side {
            let global_row = top + row as i64;
            if !(0..world).contains(&global_row) {
                continue;
            }
            let (block_row, inner_row) = (global_row >> BLOCK_SHIFT, global_row & BLOCK_MASK);
            let mut column = 0;
            while column < side {
                let global = (left + column as i64).rem_euclid(world);
                let inner = (global & BLOCK_MASK) as usize;
                // The rest of this block's row, or of the window.
                let run = (BLOCK - inner).min(side - column);
                if let Some(block) = blocks.get((block_row, global >> BLOCK_SHIFT)) {
                    let base = inner_row as usize * BLOCK;
                    for offset in 0..run {
                        let at = (base + inner + offset) * bands;
                        let values = &block[at..at + bands];
                        if values.iter().any(|value| !value.is_nan()) {
                            visit(row, column + offset, values);
                        }
                    }
                }
                column += run;
            }
        }
    }
}

/// The basic columns and rows the product's cells reach.
fn grid_extent(product: &PreparedProduct) -> (usize, usize) {
    let cells = || {
        product
            .chunks
            .iter()
            .flat_map(|(_, tile)| tile.point_positions.iter().zip(&tile.point_powers))
    };
    let max_x = cells()
        .map(|(&(x, _), &power)| usize::from(x) + (1 << power))
        .max()
        .unwrap_or(0);
    let max_y = cells()
        .map(|(&(_, y), &power)| usize::from(y) + (1 << power))
        .max()
        .unwrap_or(0);
    (max_x, max_y)
}

/// One axis of a level: the pixel each basic column or row belongs to, and
/// how many basic columns or rows each pixel spans - its share of a pixel's
/// capacity, counted as [`super::raster::aggregate_pixels`] counts it.
///
/// Each basic cell belongs to the pixel its centre falls in. The rules are
/// those of `aggregate_pixels`, evaluated over the whole world: a pixel edge
/// `t` in grid units claims the cells from `ceil(t)` on for columns, and from
/// `floor(t) + 1` on for rows, which run the other way.
struct Axis {
    owner: Vec<i64>,
    span: Vec<u64>,
}

impl Axis {
    fn span(&self, pixel: i64) -> u64 {
        usize::try_from(pixel)
            .ok()
            .and_then(|pixel| self.span.get(pixel))
            .copied()
            .unwrap_or(0)
    }

    /// Longitude names the same meridian three ways; a pixel takes its span
    /// from the naming that reaches the product's columns there, as a tile
    /// does.
    fn columns(grid: &GridSpec, level: u8, len: usize) -> Self {
        let world = (1u64 << level) as f64;
        let lng_grid =
            |pixel: f64| (pixel / world * 360.0 - 180.0 - grid.lng_0) * f64::from(grid.lng_denom);
        let turn = 360.0 * f64::from(grid.lng_denom);
        let mut owner = vec![i64::MIN; len];
        let mut span = vec![0; 1 << level];
        for shift in [0.0, turn, -turn] {
            let mut low = (lng_grid(0.0) + shift).ceil() as i64;
            for column in 0..(1i64 << level) {
                let high = (lng_grid(column as f64 + 1.0) + shift).ceil() as i64;
                let reached = low.max(0)..high.min(len as i64);
                if !reached.is_empty() {
                    span[column as usize] = (high - low) as u64;
                }
                for x in reached {
                    owner[x as usize] = column;
                }
                low = high;
            }
        }
        Self { owner, span }
    }

    fn rows(grid: &GridSpec, level: u8, len: usize) -> Self {
        let world = (1u64 << level) as f64;
        let lat_grid = |pixel: f64| {
            let latitude = web_mercator_to_lnglat(0.0, pixel / world).1;
            (latitude - grid.lat_0) * f64::from(grid.lat_denom)
        };
        let mut owner = vec![i64::MIN; len];
        let mut span = vec![0; 1 << level];
        let mut north = lat_grid(0.0).floor() as i64 + 1;
        for row in 0..(1i64 << level) {
            let south = lat_grid(row as f64 + 1.0).floor() as i64 + 1;
            span[row as usize] = (north - south).max(0) as u64;
            for y in south.max(0)..north.min(len as i64) {
                owner[y as usize] = row;
            }
            north = south;
        }
        Self { owner, span }
    }
}

/// The partial statistics of level `level`, gathered from the cells.
fn gather(
    product: &PreparedProduct,
    level: u8,
    (max_x, max_y): (usize, usize),
    every_band: bool,
) -> Blocks<Partial> {
    let spec = &product.spec;
    let bands = spec.band_specs.len();
    let reducer = spec.reducer;
    let column_of = Axis::columns(&spec.grid_spec, level, max_x).owner;
    let row_of = Axis::rows(&spec.grid_spec, level, max_y).owner;

    // Every cell once: the copy in the chunk that holds its centre. The cells
    // of a chunk run in Hilbert order, so consecutive ones mostly share a
    // block, and the last one found is kept at hand.
    let mut level = Blocks::<Partial>::default();
    let mut last_key = (i64::MIN, i64::MIN);
    let mut last_block = 0;
    for (_, tile) in &product.chunks {
        for (index, &(x, y)) in tile.point_positions.iter().enumerate() {
            if tile.point_homes[index] != (0, 0) {
                continue;
            }
            let present = |band: usize| tile.bands[band].values[index].get().is_some();
            if if every_band {
                !(0..bands).all(present)
            } else {
                !(0..bands).any(present)
            } {
                continue;
            }
            let width = 1usize << tile.point_powers[index];
            let (x, y) = (usize::from(x), usize::from(y));
            for basic_y in y..y + width {
                let Some(&row) = row_of.get(basic_y).filter(|&&row| row != i64::MIN) else {
                    continue;
                };
                for basic_x in x..x + width {
                    let Some(&column) =
                        column_of.get(basic_x).filter(|&&column| column != i64::MIN)
                    else {
                        continue;
                    };
                    let key = (row >> BLOCK_SHIFT, column >> BLOCK_SHIFT);
                    if key != last_key {
                        last_key = key;
                        let next = level.blocks.len();
                        last_block = *level.index.entry(key).or_insert(next);
                        if last_block == next {
                            level
                                .blocks
                                .push((key, vec![Partial::default(); BLOCK * BLOCK * bands]));
                        }
                    }
                    let block = &mut level.blocks[last_block].1;
                    let at = ((row & BLOCK_MASK) as usize * BLOCK + (column & BLOCK_MASK) as usize)
                        * bands;
                    for band in 0..bands {
                        if let Some(value) = tile.bands[band].values[index].get() {
                            block[at + band].add(value, 1, reducer);
                        }
                    }
                }
            }
        }
    }
    level
}

/// The level one coarser than `finer`: each pixel merges the four under it.
/// A block's four children fill one quadrant each.
fn halve(finer: &Blocks<Partial>, bands: usize, reducer: Reducer) -> Blocks<Partial> {
    let mut parents = finer
        .index
        .keys()
        .map(|&(row, column)| (row >> 1, column >> 1))
        .collect::<Vec<_>>();
    parents.sort_unstable();
    parents.dedup();
    Blocks::from_blocks(
        parents
            .into_par_iter()
            .map(|parent| {
                let mut out = vec![Partial::default(); BLOCK * BLOCK * bands];
                for (down, across) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let Some(child) = finer.get((2 * parent.0 + down, 2 * parent.1 + across))
                    else {
                        continue;
                    };
                    let (quadrant_row, quadrant_column) =
                        (down as usize * BLOCK / 2, across as usize * BLOCK / 2);
                    for row in 0..BLOCK {
                        let out_row = quadrant_row + row / 2;
                        for column in 0..BLOCK {
                            let from = (row * BLOCK + column) * bands;
                            let to = (out_row * BLOCK + quadrant_column + column / 2) * bands;
                            for band in 0..bands {
                                out[to + band].merge(child[from + band], reducer);
                            }
                        }
                    }
                }
                (parent, out)
            })
            .collect(),
    )
}
