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
//! A percentile cannot be recovered from the percentiles below it, so for one
//! a pixel keeps how much of its area each stored value covers instead
//! ([`Histograms`]): the four under it add up the same way, and the value at
//! the rank is picked once the level is reached. The values are the raw
//! integers the grid stores, so nothing is binned or approximated.
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

/// Blocks of `BLOCK` × `BLOCK` pixels, each holding the state of `bands`
/// slots per pixel row by row, found by their global block row and column.
struct Blocks<B> {
    index: HashMap<(i64, i64), usize>,
    blocks: Vec<((i64, i64), B)>,
}

impl<B> Default for Blocks<B> {
    fn default() -> Self {
        Self {
            index: HashMap::default(),
            blocks: Vec::new(),
        }
    }
}

impl<B> Blocks<B> {
    fn get(&self, key: (i64, i64)) -> Option<&B> {
        self.index.get(&key).map(|&at| &self.blocks[at].1)
    }

    fn from_blocks(blocks: Vec<((i64, i64), B)>) -> Self {
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
    levels: Vec<Blocks<Vec<f64>>>,
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
        let extent = grid_extent(product);
        let levels = if let Reducer::Percentile(_) = reducer {
            let finest_level = gather(
                product,
                finest,
                extent,
                every_band,
                Vec::new,
                |block: &mut Vec<(u32, i32, u64)>, slot, value| {
                    // A cell covers its basic cells in runs along a row, and
                    // most of a run lands in one pixel.
                    match block.last_mut() {
                        Some(last) if last.0 == slot as u32 && last.1 == value => last.2 += 1,
                        _ => block.push((slot as u32, value, 1)),
                    }
                },
            );
            let finest_level = Blocks::from_blocks(
                finest_level
                    .blocks
                    .into_par_iter()
                    .map(|(key, block)| {
                        (key, Histograms::from_unsorted(block, BLOCK * BLOCK * bands))
                    })
                    .collect(),
            );
            finish_levels(
                product,
                (coarsest, finest),
                extent,
                finest_level,
                |histograms: &Histograms, slot, capacity, fill, _: &mut ()| {
                    super::reduce::finish_histogram(histograms.get(slot), reducer, capacity, fill)
                },
                |finer| halve_histograms(finer, bands),
            )
        } else {
            let finest_level = gather(
                product,
                finest,
                extent,
                every_band,
                || vec![Partial::default(); BLOCK * BLOCK * bands],
                |block: &mut Vec<Partial>, slot, value| block[slot].add(value, 1, reducer),
            );
            finish_levels(
                product,
                (coarsest, finest),
                extent,
                finest_level,
                |block: &Vec<Partial>, slot, capacity, fill, _: &mut ()| {
                    block[slot].finish(reducer, capacity, fill)
                },
                |finer| halve(finer, bands, reducer),
            )
        };
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

/// Finishes every level from `finest` down to `coarsest`, halving the partial
/// state between them, and returns the levels coarsest first. A level's state
/// lives only until the next one is made from it.
fn finish_levels<T: Sync + Send, S: Default>(
    product: &PreparedProduct,
    (coarsest, finest): (u8, u8),
    (max_x, max_y): (usize, usize),
    mut state: Blocks<T>,
    finish: impl Fn(&T, usize, u64, Option<i32>, &mut S) -> Option<f64> + Sync,
    halve: impl Fn(&Blocks<T>) -> Blocks<T>,
) -> Vec<Blocks<Vec<f64>>> {
    let spec = &product.spec;
    let bands = spec.band_specs.len();
    let mut levels = Vec::new();
    for level in (coarsest..=finest).rev() {
        let columns = Axis::columns(&spec.grid_spec, level, max_x);
        let rows = Axis::rows(&spec.grid_spec, level, max_y);
        levels.push(Blocks::from_blocks(
            state
                .blocks
                .par_iter()
                .map_init(S::default, |scratch, &(key, ref block)| {
                    let mut out = vec![f64::NAN; BLOCK * BLOCK * bands];
                    for row in 0..BLOCK {
                        let height = rows.span(key.0 * BLOCK as i64 + row as i64);
                        for column in 0..BLOCK {
                            let width = columns.span(key.1 * BLOCK as i64 + column as i64);
                            let at = (row * BLOCK + column) * bands;
                            for band in 0..bands {
                                if let Some(value) = finish(
                                    block,
                                    at + band,
                                    height * width,
                                    spec.fill[band],
                                    scratch,
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
            state = halve(&state);
        }
    }
    levels.reverse();
    levels
}

/// The stored values under each pixel of a block and the area each covers,
/// for a percentile: slot `s` (pixel × bands + band) holds
/// `entries[offsets[s]..offsets[s + 1]]`, sorted by value, each value once.
struct Histograms {
    offsets: Vec<u32>,
    entries: Vec<(i32, u64)>,
}

impl Histograms {
    fn get(&self, slot: usize) -> &[(i32, u64)] {
        &self.entries[self.offsets[slot] as usize..self.offsets[slot + 1] as usize]
    }

    /// From `(slot, value, weight)` in any order, repeats allowed.
    fn from_unsorted(items: Vec<(u32, i32, u64)>, slots: usize) -> Self {
        // Bucketed by slot first, which leaves a few values to order in each.
        let mut offsets = vec![0u32; slots + 1];
        for &(slot, ..) in &items {
            offsets[slot as usize + 1] += 1;
        }
        for slot in 0..slots {
            offsets[slot + 1] += offsets[slot];
        }
        let mut next = offsets.clone();
        let mut bucketed = vec![(0, 0); items.len()];
        for (slot, value, weight) in items {
            bucketed[next[slot as usize] as usize] = (value, weight);
            next[slot as usize] += 1;
        }
        let mut entries: Vec<(i32, u64)> = Vec::with_capacity(bucketed.len());
        for slot in 0..slots {
            let bucket = &mut bucketed[offsets[slot] as usize..offsets[slot + 1] as usize];
            bucket.sort_unstable_by_key(|&(value, _)| value);
            offsets[slot] = entries.len() as u32;
            let first = entries.len();
            for &(value, weight) in bucket.iter() {
                match entries[first..].last_mut() {
                    Some(last) if last.0 == value => last.1 += weight,
                    _ => entries.push((value, weight)),
                }
            }
        }
        offsets[slots] = entries.len() as u32;
        entries.shrink_to_fit();
        Self { offsets, entries }
    }
}

/// Merges sorted `(value, weight)` lists into `out`, each value once with its
/// weights added.
fn merge_sorted(lists: &mut [&[(i32, u64)]], out: &mut Vec<(i32, u64)>) {
    loop {
        let Some(value) = lists
            .iter()
            .filter_map(|list| list.first())
            .map(|e| e.0)
            .min()
        else {
            return;
        };
        let mut weight = 0;
        for list in lists.iter_mut() {
            if let Some(&(head, add)) = list.first()
                && head == value
            {
                weight += add;
                *list = &list[1..];
            }
        }
        out.push((value, weight));
    }
}

/// [`halve`] for histograms: each pixel's values are those of the four under
/// it, with the weights of a value they share added.
fn halve_histograms(finer: &Blocks<Histograms>, bands: usize) -> Blocks<Histograms> {
    let mut parents = finer
        .index
        .keys()
        .map(|&(row, column)| (row >> 1, column >> 1))
        .collect::<Vec<_>>();
    parents.sort_unstable();
    parents.dedup();
    const HALF: usize = BLOCK / 2;
    Blocks::from_blocks(
        parents
            .into_par_iter()
            .map(|parent| {
                let children = [(0, 0), (0, 1), (1, 0), (1, 1)]
                    .map(|(down, across)| finer.get((2 * parent.0 + down, 2 * parent.1 + across)));
                let bound = children.iter().flatten().map(|c| c.entries.len()).sum();
                let mut offsets = Vec::with_capacity(BLOCK * BLOCK * bands + 1);
                let mut entries = Vec::with_capacity(bound);
                for row in 0..BLOCK {
                    for column in 0..BLOCK {
                        let child = children[(row / HALF) * 2 + column / HALF];
                        let (row, column) = (2 * (row % HALF), 2 * (column % HALF));
                        for band in 0..bands {
                            offsets.push(entries.len() as u32);
                            let Some(child) = child else { continue };
                            let slot =
                                |r: usize, c: usize| child.get((r * BLOCK + c) * bands + band);
                            merge_sorted(
                                &mut [
                                    slot(row, column),
                                    slot(row, column + 1),
                                    slot(row + 1, column),
                                    slot(row + 1, column + 1),
                                ],
                                &mut entries,
                            );
                        }
                    }
                }
                offsets.push(entries.len() as u32);
                entries.shrink_to_fit();
                (parent, Histograms { offsets, entries })
            })
            .collect(),
    )
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

/// The partial state of level `level`, gathered from the cells: `add` takes
/// a block, the slot of a pixel's band in it and one basic cell's value.
fn gather<T>(
    product: &PreparedProduct,
    level: u8,
    (max_x, max_y): (usize, usize),
    every_band: bool,
    new_block: impl Fn() -> T,
    mut add: impl FnMut(&mut T, usize, i32),
) -> Blocks<T> {
    let spec = &product.spec;
    let bands = spec.band_specs.len();
    let column_of = Axis::columns(&spec.grid_spec, level, max_x).owner;
    let row_of = Axis::rows(&spec.grid_spec, level, max_y).owner;

    // Every cell once: the copy in the chunk that holds its centre. The cells
    // of a chunk run in Hilbert order, so consecutive ones mostly share a
    // block, and the last one found is kept at hand.
    let mut level = Blocks::<T>::default();
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
                            level.blocks.push((key, new_block()));
                        }
                    }
                    let block = &mut level.blocks[last_block].1;
                    let at = ((row & BLOCK_MASK) as usize * BLOCK + (column & BLOCK_MASK) as usize)
                        * bands;
                    for band in 0..bands {
                        if let Some(value) = tile.bands[band].values[index].get() {
                            add(block, at + band, value);
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
fn halve(finer: &Blocks<Vec<Partial>>, bands: usize, reducer: Reducer) -> Blocks<Vec<Partial>> {
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
