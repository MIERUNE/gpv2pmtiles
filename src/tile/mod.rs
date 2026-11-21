mod detail;
mod mvt;
mod raster;
mod reduce;

pub(crate) use detail::DetailSchedule;
pub(crate) use reduce::Reducer;

pub(crate) use raster::{
    ColorEncoding, RangeReport, RasterLayout, TILE_BUFFER as RASTER_TILE_BUFFER,
    TILE_SIZES as RASTER_TILE_SIZES,
};
mod rect_union;

use anyhow::Result;
use fast_mvt::MvtValue;
use gpv_products::model::Aggregation;
use rayon::prelude::*;

use fast_hilbert::{h2xy, xy2h};

use crate::{
    geo::{LngLat, LngLatBox, web_mercator_to_lnglat},
    model::{CompactOptI32, PreparedProduct},
    quantize::{BandOmit, BandQuantize},
};

pub(crate) type Zxy = (u8, u32, u32);
type PointMap<K, V> = foldhash::HashMap<K, V>;

#[derive(Debug, Clone, Copy)]
struct TileBounds {
    mx1: f64,
    my1: f64,
    mx2: f64,
    my2: f64,
}

pub(crate) struct TileContext {
    zxy: Zxy,
    bounds: TileBounds,
    geographic_bounds: LngLatBox,
    wrapped_geographic_bounds: LngLatBox,
}

impl TileContext {
    pub(crate) fn new(zxy: Zxy) -> Self {
        let (z, x, y) = zxy;
        let bounds = TileBounds {
            mx1: x as f64 / (1 << z) as f64,
            my1: y as f64 / (1 << z) as f64,
            mx2: (x + 1) as f64 / (1 << z) as f64,
            my2: (y + 1) as f64 / (1 << z) as f64,
        };
        let buffer = (bounds.mx2 - bounds.mx1) * (f64::from(CONTEXT_BUFFER) / 256.0);
        let (lng1, lat1) = web_mercator_to_lnglat(bounds.mx1 - buffer, bounds.my2 + buffer);
        let (lng2, lat2) = web_mercator_to_lnglat(bounds.mx2 + buffer, bounds.my1 - buffer);

        Self {
            zxy,
            bounds,
            geographic_bounds: LngLatBox::new(LngLat::new(lng1, lat1), LngLat::new(lng2, lat2)),
            wrapped_geographic_bounds: LngLatBox::new(
                LngLat::new(lng1 + 360.0, lat1),
                LngLat::new(lng2 + 360.0, lat2),
            ),
        }
    }
}

#[derive(Default, PartialEq, Eq, Debug, PartialOrd)]
struct Point {
    values: [CompactOptI32; 4],
    counts: [i32; 4],
    power: u8,
}

struct BandScale<'a> {
    name: &'a str,
    scaling: bool,
    reference: f64,
    binary_scale: f64,
    decimal_scale: f64,
    /// When set, the incoming value is a class index rather than a raw value.
    quantize: Option<&'a BandQuantize>,
}

impl BandScale<'_> {
    /// Converts a raw band value into an MVT attribute value.
    ///
    /// The MVT value type is chosen from the band spec alone, never from the
    /// value: a scaled band always yields `double`, an unscaled one always
    /// yields `sint`. Mixing types within a single attribute is legal in MVT,
    /// but consumers that give each attribute one column type - MapLibre Tiles
    /// among them - cannot represent it, and can drop or corrupt the values.
    /// Since `scaling` is derived from the band spec, the type stays the same
    /// across every feature, tile and zoom level of a layer.
    ///
    /// The scaling is computed in `f64` and emitted as `double` so that the
    /// whole `i32` range of raw values survives: `f32` only holds integers
    /// exactly up to 2^24, and an `i32` output would instead saturate once the
    /// scaled value leaves the `i32` range.
    ///
    /// Note that `MvtValue::from` must not be used for the integer case: it
    /// picks `uint` or `sint` depending on the sign, so a band spanning zero
    /// would mix both.
    /// Whether the band's values are class indices rather than measurements.
    #[inline]
    fn is_quantized(&self) -> bool {
        self.quantize.is_some()
    }

    /// The physical value a **raw GRIB reading** ends up as in a tile.
    ///
    /// Quantization happens between the two, so this has to apply it rather
    /// than skip it: [`Self::physical`] expects a value that has already been
    /// through it and would read a raw value as a class index, indexing the
    /// class table out of bounds.
    #[inline]
    fn source_physical(&self, raw: i32) -> f64 {
        match self.quantize {
            Some(quantize) => quantize.output(quantize.class_of(raw)),
            None => self.physical(raw),
        }
    }

    /// The value in the band's physical unit, as the raster encoders need it.
    ///
    /// Takes a **tile** value: one that has already been quantized, if the band
    /// is quantized at all. Use [`Self::source_physical`] for a raw reading.
    /// Mirrors the branches of [`Self::tag_value`], which chooses an MVT type
    /// for the same number.
    #[inline]
    fn physical(&self, value: i32) -> f64 {
        if let Some(quantize) = self.quantize {
            quantize.output(value)
        } else if self.scaling {
            (self.reference + f64::from(value) * self.binary_scale) * self.decimal_scale
        } else {
            f64::from(value)
        }
    }

    #[inline]
    fn tag_value(&self, value: i32) -> MvtValue {
        if let Some(quantize) = self.quantize {
            // `value` is a class index here; the scaling was already applied
            // when the boundaries were resolved.
            let output = quantize.output(value);
            return if quantize.integral_outputs() {
                MvtValue::SInt(output as i64)
            } else {
                MvtValue::Double(output)
            };
        }
        if self.scaling {
            MvtValue::Double(
                (self.reference + (value as f64) * self.binary_scale) * self.decimal_scale,
            )
        } else {
            MvtValue::SInt(value as i64)
        }
    }
}

#[inline]
fn aggregate_band(values: &[CompactOptI32], aggregation: Aggregation) -> (CompactOptI32, i32) {
    match aggregation {
        Aggregation::Max => (CompactOptI32::new(values.iter().flatten().max()), 0),
        Aggregation::Min => (CompactOptI32::new(values.iter().flatten().min()), 0),
        Aggregation::RoughAvg => {
            let mut sum = None;
            let mut count = 0usize;
            for value in values.iter().flatten() {
                sum = Some(match sum {
                    Some(sum) => sum + value,
                    None => value,
                });
                count += 1;
            }
            (CompactOptI32::new(sum), count as i32)
        }
        Aggregation::BitOr => (
            CompactOptI32::new(
                values
                    .iter()
                    .flatten()
                    .fold(None, |acc, value| Some(acc.unwrap_or(0) | value)),
            ),
            0,
        ),
    }
}

fn merge_point(
    points: &mut PointMap<(u32, u32), Point>,
    key: (u32, u32),
    incoming: Point,
    aggregation: Aggregation,
    band_count: usize,
) {
    use std::collections::hash_map::Entry;

    match points.entry(key) {
        Entry::Vacant(entry) => {
            entry.insert(incoming);
        }
        Entry::Occupied(mut entry) => {
            let point = entry.get_mut();
            match aggregation {
                Aggregation::Max => {
                    for index in 0..band_count {
                        if let Some(value) = incoming.values[index].get() {
                            point.values[index] = CompactOptI32::new(Some(
                                point.values[index].unwrap_or(value).max(value),
                            ));
                        }
                    }
                }
                Aggregation::Min => {
                    for index in 0..band_count {
                        if let Some(value) = incoming.values[index].get() {
                            point.values[index] = CompactOptI32::new(Some(
                                point.values[index].unwrap_or(value).min(value),
                            ));
                        }
                    }
                }
                Aggregation::RoughAvg => {
                    for index in 0..band_count {
                        if let Some(value) = incoming.values[index].get() {
                            point.values[index] =
                                CompactOptI32::new(Some(point.values[index].unwrap_or(0) + value));
                        }
                        point.counts[index] += incoming.counts[index];
                    }
                }
                Aggregation::BitOr => {
                    for index in 0..band_count {
                        if let Some(value) = incoming.values[index].get() {
                            point.values[index] =
                                CompactOptI32::new(Some(point.values[index].unwrap_or(0) | value));
                        }
                    }
                }
            }
            point.power = point.power.max(incoming.power);
        }
    }
}

/// Clears omitted band values and removes points with no remaining attributes.
fn apply_omits(
    points: &mut PointMap<(u32, u32), Point>,
    omits: &[Option<BandOmit>],
    band_count: usize,
) -> bool {
    if !omits.iter().any(Option::is_some) {
        return false;
    }
    // Clearing an omitted value and dropping a point left with no values happen
    // in one pass: the retained set only depends on values this closure sees.
    points.retain(|_, point| {
        for (index, omit) in omits.iter().enumerate() {
            if let Some(omit) = omit {
                point.values[index] = point.values[index].map(|value| {
                    if omit.contains(value) {
                        CompactOptI32::NONE
                    } else {
                        CompactOptI32::new(Some(value))
                    }
                });
            }
        }
        point.values[..band_count]
            .iter()
            .any(|value| value.get().is_some())
    });
    points.is_empty()
}

pub(crate) fn zxy_to_chunk_id_range(base_z: u8, zxy: Zxy) -> (u64, u64) {
    let (z, x, y) = zxy;
    if z < base_z {
        let scale = 1 << (base_z - z);
        let begin = xy2h(x, y, z) * scale * scale;
        (begin, begin + scale * scale)
    } else {
        let begin = xy2h(x >> (z - base_z), y >> (z - base_z), base_z);
        (begin, begin + 1)
    }
}

/// The chunks a tile reads, as a rectangle of base-zoom tiles.
///
/// Longitude wraps, so the columns are a run that may cross the antimeridian:
/// `columns` of them from `first_column`, modulo the width of the zoom.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ChunkSquare {
    across: u32,
    first_column: u32,
    columns: u32,
    rows: (u32, u32),
}

impl ChunkSquare {
    /// The chunks [`source_chunks`] returns for a tile.
    fn read_by(base_z: u8, zxy: Zxy, output: TileOutput) -> Self {
        let (z, x, y) = zxy;
        let across = 1u32 << base_z;
        // A tile and its neighbours when the margin is read.
        let reach = i64::from(matches!(output, TileOutput::Raster(_)));
        let last_tile = (1i64 << z) - 1;
        let (column_begin, column_end) = (i64::from(x) - reach, i64::from(x) + reach);
        let (row_begin, row_end) = (
            (i64::from(y) - reach).max(0),
            (i64::from(y) + reach).min(last_tile),
        );
        // Tile edges at `z` in chunk units: below the base zoom a tile covers
        // several chunks, above it a chunk covers several tiles.
        let to_chunks = |begin: i64, end: i64| -> (i64, i64) {
            if z <= base_z {
                let shift = base_z - z;
                (begin << shift, ((end + 1) << shift) - 1)
            } else {
                let shift = z - base_z;
                (begin >> shift, end >> shift)
            }
        };
        let (first, last) = to_chunks(column_begin, column_end);
        let (top, bottom) = to_chunks(row_begin, row_end);
        Self {
            across,
            first_column: first.rem_euclid(i64::from(across)) as u32,
            columns: (last - first + 1).min(i64::from(across)) as u32,
            rows: (top as u32, bottom as u32),
        }
    }

    /// The chunk of this square nearest to `chunk`, which is `chunk` itself
    /// when the square holds it.
    ///
    /// A cell's copies cover a rectangle of chunks around its own, so when its
    /// own chunk lies outside the square, the nearest one inside is among the
    /// copies the tile reads, and taking it keeps exactly one.
    fn nearest(self, (column, row): (u32, u32)) -> (u32, u32) {
        let offset = (column + self.across - self.first_column) % self.across;
        let column = if offset < self.columns {
            column
        } else {
            let past_end = offset - (self.columns - 1);
            let before_start = self.across - offset;
            if past_end < before_start {
                (self.first_column + self.columns - 1) % self.across
            } else {
                self.first_column
            }
        };
        (column, row.clamp(self.rows.0, self.rows.1))
    }
}

/// Reduces the cells of a tile into blocks with a weighted [`Reducer`].
///
/// Every cell is counted once, through [`ChunkSquare::nearest`], and weighs
/// the base cells it spans. `keep` is told a block's corner and width and
/// decides whether it reaches the tile.
///
/// A chunk keeps its cells in Hilbert order, which visits every aligned block
/// in one stretch, so a block is a run of consecutive cells in each chunk it
/// reaches. Only the runs are gathered and sorted, not the cells; a block that
/// straddles chunks is the few runs that share its corner.
fn weighted_blocks(
    product: &PreparedProduct,
    chunks: &[&[(u64, crate::model::BaseTile)]],
    square: ChunkSquare,
    aggregation_scale: u8,
    keep: impl Fn(u32, u32, u32) -> bool + Sync,
) -> PointMap<(u32, u32), Point> {
    let base_z = product.spec.base_z;
    let band_count = product.spec.band_specs.len();
    let mask = !((1u32 << aggregation_scale) - 1);

    // A margin can name the same chunk for several neighbouring tiles.
    let mut read = chunks
        .iter()
        .flat_map(|chunk| chunk.iter())
        .collect::<Vec<_>>();
    read.sort_unstable_by_key(|(id, _)| *id);
    read.dedup_by_key(|(id, _)| *id);

    let key_of = |tile: &crate::model::BaseTile, index: usize| {
        let (x, y) = tile.point_positions[index];
        (u64::from(u32::from(x) & mask) << 32) | u64::from(u32::from(y) & mask)
    };
    // Whether this copy of a cell is the one that counts.
    let counts = |(column, row): (u32, u32), tile: &crate::model::BaseTile, index: usize| {
        let (offset_x, offset_y) = tile.point_homes[index];
        // Most copies are the cell's own, and every chunk read lies in the
        // square, so those count without asking where the nearest one is.
        if (offset_x, offset_y) == (0, 0) {
            return true;
        }
        let home = (
            (i64::from(column) + i64::from(offset_x)).rem_euclid(i64::from(square.across)) as u32,
            (i64::from(row) + i64::from(offset_y)) as u32,
        );
        square.nearest(home) == (column, row)
    };

    // (block key, index into `read`, first cell, end of the run)
    let mut runs = read
        .par_iter()
        .enumerate()
        .flat_map_iter(|(chunk, (id, tile))| {
            let corner = h2xy::<u32>(*id, base_z);
            let mut runs = Vec::new();
            let mut begin = 0;
            while begin < tile.point_positions.len() {
                let key = key_of(tile, begin);
                let mut end = begin + 1;
                while end < tile.point_positions.len() && key_of(tile, end) == key {
                    end += 1;
                }
                if (begin..end).any(|index| counts(corner, tile, index)) {
                    runs.push((key, chunk as u32, begin as u32, end as u32));
                }
                begin = end;
            }
            runs
        })
        .collect::<Vec<_>>();
    runs.par_sort_unstable_by_key(|&(key, ..)| key);

    runs.par_chunk_by(|left, right| left.0 == right.0)
        .map_init(<[reduce::BlockSamples; 4]>::default, |samples, block| {
            let key = block[0].0;
            let (grid_x, grid_y) = ((key >> 32) as u32, key as u32);
            // One pass gathers every band and the block's size together.
            samples.iter_mut().for_each(reduce::BlockSamples::clear);
            let mut power = aggregation_scale;
            for &(_, chunk, begin, end) in block {
                let (id, tile) = read[chunk as usize];
                let corner = h2xy::<u32>(*id, base_z);
                for index in begin as usize..end as usize {
                    if !counts(corner, tile, index) {
                        continue;
                    }
                    let cell_power = tile.point_powers[index];
                    power = power.max(cell_power);
                    let weight = 1 << (2 * u32::from(cell_power));
                    for (band, band_samples) in tile.bands.iter().zip(samples.iter_mut()) {
                        if let Some(raw) = band.values[index].get() {
                            band_samples.push(raw, weight);
                        }
                    }
                }
            }
            if !keep(grid_x, grid_y, 1 << power) {
                return None;
            }
            let capacity = 1u64 << (2 * u32::from(power));
            let mut values = [CompactOptI32::NONE; 4];
            for (band_index, (value, band_samples)) in values
                .iter_mut()
                .zip(samples.iter_mut())
                .enumerate()
                .take(band_count)
            {
                *value = CompactOptI32::new(band_samples.reduce(
                    product.spec.reducer,
                    capacity,
                    product.spec.fill[band_index],
                ));
            }
            values.iter().any(|value| value.get().is_some()).then_some((
                (grid_x, grid_y),
                Point {
                    values,
                    counts: [0; 4],
                    power,
                },
            ))
        })
        .flatten()
        .collect()
}

/// Measures a product's source values against the range an encoding can hold.
///
/// Wraps the band scaling rather than exposing it: the physical values are an
/// internal detail of how a tile is built.
pub(crate) fn inspect_raster_range(
    product: &PreparedProduct,
    encoding: ColorEncoding,
) -> RangeReport {
    raster::inspect_range(&product.chunks, &band_scales(product), encoding)
}

/// How each band's raw values turn into physical ones.
fn band_scales(product: &PreparedProduct) -> Vec<BandScale<'_>> {
    product
        .spec
        .band_specs
        .iter()
        .zip(&product.spec.quantize)
        .map(|(band, quantize)| BandScale {
            name: &band.name,
            quantize: quantize.as_ref(),
            scaling: band.binary_scale != 0
                || band.reference_value != 0.0
                || band.decimal_scale != 0,
            reference: band.reference_value as f64,
            binary_scale: 2f64.powi(band.binary_scale as i32),
            decimal_scale: 10f64.powi(-band.decimal_scale as i32),
        })
        .collect()
}

/// The chunks a tile has to read to be rendered.
///
/// A raster tile carries a margin of its neighbours' pixels so that a client
/// filtering across the boundary does not clamp at the edge, and the values in
/// that margin come from points outside the tile. Reading only the tile's own
/// chunks leaves the margin empty, which puts a one-pixel line of "no data"
/// along every tile boundary - the opposite of what the margin is for.
///
/// The surrounding tiles are read whole and the points outside the margin are
/// dropped by the bounds test in [`make_layer`], which is cheaper than working
/// out which chunks the margin actually touches.
fn source_chunks(
    product: &PreparedProduct,
    zxy: Zxy,
    output: TileOutput,
) -> Vec<&[(u64, crate::model::BaseTile)]> {
    let (z, x, y) = zxy;
    let range = |zxy| {
        let (begin, end) = zxy_to_chunk_id_range(product.spec.base_z, zxy);
        product.chunks_in_range(begin, end)
    };
    if !matches!(output, TileOutput::Raster(_)) {
        return vec![range(zxy)];
    }

    let last = (1u32 << z) - 1;
    let mut chunks = Vec::with_capacity(9);
    for dy in -1i64..=1 {
        for dx in -1i64..=1 {
            // Longitude wraps; latitude does not.
            let Some(neighbour_y) = y.checked_add_signed(dy as i32).filter(|y| *y <= last) else {
                continue;
            };
            let neighbour_x = (i64::from(x) + dx).rem_euclid(i64::from(last) + 1) as u32;
            let chunk = range((z, neighbour_x, neighbour_y));
            if !chunk.is_empty() {
                chunks.push(chunk);
            }
        }
    }
    chunks
}

/// What a tile is rendered as.
///
/// MVT layers of several products concatenate into one tile, which is how a
/// forecast sequence becomes one archive. Images cannot be concatenated, so a
/// raster archive holds a single product; [`crate::archive`] enforces that.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum TileOutput {
    Mvt(MvtLayout),
    Raster(RasterLayout),
}

/// The coordinate space of an MVT tile.
///
/// A client does not care which is used, but matching them to an existing
/// tileset lets the tiles be compared, or substituted, feature for feature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MvtLayout {
    /// Units across the tile.
    pub(crate) extent: u32,
    /// Geometry kept beyond each edge, in 1/256ths of the tile width - the
    /// pixels of a 256px tile, as tippecanoe counts them.
    pub(crate) buffer: u32,
    /// How coarsely cells are merged at each zoom; see [`detail`].
    pub(crate) detail: DetailSchedule,
}

impl Default for MvtLayout {
    fn default() -> Self {
        Self {
            extent: 4096,
            buffer: 2,
            detail: DetailSchedule::default(),
        }
    }
}

/// The margin [`TileContext`] reads around every tile, in the same units as
/// [`MvtLayout::buffer`].
const CONTEXT_BUFFER: u32 = 2;

pub(crate) fn make_layer(
    tile_context: &TileContext,
    product: &PreparedProduct,
    layer_name: &str,
    output: TileOutput,
) -> Result<Option<Vec<u8>>> {
    let (z, _, _) = tile_context.zxy;
    let (chunk_begin, chunk_end) = zxy_to_chunk_id_range(product.spec.base_z, tile_context.zxy);
    if product.chunks_in_range(chunk_begin, chunk_end).is_empty() {
        return Ok(None);
    }
    let chunks = source_chunks(product, tile_context.zxy, output);

    let maximum_detail_zoom = (product.spec.grid_spec.lat_denom * 360.0 * 2.0 / 512.0)
        .log2()
        .round() as u8;

    // A raster tile interpolates, so it needs the grid points bracketing every
    // pixel it writes - including the ones in its margin, which sit outside the
    // tile. The margin `TileContext` carries is a fraction of the tile, and at
    // the deeper zooms that is narrower than one grid cell, which left the
    // outermost point of the bracket behind and made a seam disagree by a code.
    let reach = match output {
        TileOutput::Raster(_) => {
            let grid = &product.spec.grid_spec;
            (
                2.0 / f64::from(grid.lng_denom),
                2.0 / f64::from(grid.lat_denom).abs(),
            )
        }
        // A buffer wider than the one the context reads needs the cells under
        // the rest of it. Degrees of latitude span no more of the mercator
        // tile than the same degrees of longitude, so the longitude width
        // covers both.
        TileOutput::Mvt(layout) => {
            let excess = layout.buffer.saturating_sub(CONTEXT_BUFFER);
            let degrees = 360.0 / f64::from(1u32 << z) * f64::from(excess) / 256.0;
            (degrees, degrees)
        }
    };
    let geographic_bounds = tile_context.geographic_bounds.expanded(reach.0, reach.1);
    let wrapped_geographic_bounds = tile_context
        .wrapped_geographic_bounds
        .expanded(reach.0, reach.1);

    // A weighted statistic of a block that is a single cell is that cell's
    // value, and the cells of a tile do not overlap, so where nothing is
    // merged the maximum gives the same answer without weighing anything.
    let aggregation = match product.spec.reducer {
        Reducer::Legacy(aggregation) => aggregation,
        _ => Aggregation::Max,
    };
    let band_count = product.spec.band_specs.len();
    let aggregation_scale = match output {
        TileOutput::Mvt(layout) => detail::mvt_aggregation_scale(
            maximum_detail_zoom,
            z,
            layout.detail.at(z),
            (
                product.spec.grid_spec.lng_denom,
                product.spec.grid_spec.lat_denom,
            ),
            layout.extent,
        ),
        // A smaller tile reaches the same pixels a zoom later, and is merged
        // as the larger one is there; see `RasterLayout::zoom_offset`.
        TileOutput::Raster(layout) => (i32::from(maximum_detail_zoom) - i32::from(z)
            + i32::from(layout.zoom_offset()))
        .max(0) as u8,
    };
    let aggregation_width = 1u32 << aggregation_scale;
    let aggregation_mask = !(aggregation_width - 1);
    let grid = &product.spec.grid_spec;

    let ingest = |points: &mut PointMap<(u32, u32), Point>, tile: &crate::model::BaseTile| {
        debug_assert_eq!(tile.point_positions.len(), tile.point_powers.len());
        let group_key = |index: usize| {
            let (x, y) = tile.point_positions[index];
            (
                u32::from(x) & aggregation_mask,
                u32::from(y) & aggregation_mask,
                tile.point_powers[index],
            )
        };

        let mut begin = 0;
        while begin < tile.point_positions.len() {
            let group_begin = begin;
            let (grid_x, grid_y, point_power) = group_key(begin);
            let mut end = begin + 1;
            while end < tile.point_positions.len()
                && group_key(end) == (grid_x, grid_y, point_power)
            {
                end += 1;
            }
            begin = end;
            let power = point_power.max(aggregation_scale);
            let width = 1 << power;

            let lng1 = grid.lng_0 + (grid_x as f64 - 0.5) / grid.lng_denom as f64;
            let lng2 = grid.lng_0 + ((grid_x + width) as f64 - 0.5) / grid.lng_denom as f64;
            let lat2 = grid.lat_0 + (grid_y as f64 - 0.5) / grid.lat_denom as f64;
            let lat1 = grid.lat_0 + ((grid_y + width) as f64 - 0.5) / grid.lat_denom as f64;
            let point_box = LngLatBox::new(LngLat::new(lng1, lat1), LngLat::new(lng2, lat2));
            if !geographic_bounds.intersects_box(&point_box)
                && (lng2 <= 180.0 || !wrapped_geographic_bounds.intersects_box(&point_box))
            {
                continue;
            }

            let mut values = [CompactOptI32::NONE; 4];
            let mut counts = [0; 4];
            for (band_index, band) in tile.bands.iter().enumerate() {
                (values[band_index], counts[band_index]) =
                    aggregate_band(&band.values[group_begin..end], aggregation);
            }

            merge_point(
                points,
                (grid_x, grid_y),
                Point {
                    values,
                    counts,
                    power,
                },
                aggregation,
                band_count,
            );
        }
    };

    let mut points = if product.spec.reducer.is_weighted() && aggregation_scale > 0 {
        weighted_blocks(
            product,
            &chunks,
            ChunkSquare::read_by(product.spec.base_z, tile_context.zxy, output),
            aggregation_scale,
            |grid_x, grid_y, width| {
                let lng1 = grid.lng_0 + (grid_x as f64 - 0.5) / grid.lng_denom as f64;
                let lng2 = grid.lng_0 + ((grid_x + width) as f64 - 0.5) / grid.lng_denom as f64;
                let lat2 = grid.lat_0 + (grid_y as f64 - 0.5) / grid.lat_denom as f64;
                let lat1 = grid.lat_0 + ((grid_y + width) as f64 - 0.5) / grid.lat_denom as f64;
                let point_box = LngLatBox::new(LngLat::new(lng1, lat1), LngLat::new(lng2, lat2));
                geographic_bounds.intersects_box(&point_box)
                    || (lng2 > 180.0 && wrapped_geographic_bounds.intersects_box(&point_box))
            },
        )
    } else if aggregation_scale == 0 {
        // The group key is the grid cell itself here, so cells stay distinct
        // across chunks; only cells lying on a chunk boundary repeat, in both
        // neighbours. Sizing the map to the input count keeps it from ever
        // rehashing, and one map avoids both the per-thread copies a parallel
        // fold would build and the merge that would follow. The tile itself is
        // already processed in parallel with its siblings.
        let capacity = chunks
            .iter()
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|(_, tile)| tile.point_positions.len())
                    .sum::<usize>()
            })
            .sum();
        let mut points = PointMap::with_capacity_and_hasher(capacity, <_>::default());
        for chunk in &chunks {
            for (_, tile) in chunk.iter() {
                ingest(&mut points, tile);
            }
        }
        points
    } else {
        chunks
            .par_iter()
            .map(|chunk| {
                let point_count = chunk
                    .iter()
                    .map(|(_, tile)| tile.point_positions.len())
                    .sum::<usize>();
                // Dense coverage packs 4^scale cells into one group; sparse
                // data keeps them apart. Sizing to the dense count only
                // under-estimates then, which costs a rehash rather than
                // correctness, and never over-allocates past the input.
                let capacity = (point_count >> (2 * aggregation_scale)).max(16);
                let mut points = PointMap::with_capacity_and_hasher(capacity, <_>::default());
                for (_, tile) in chunk.iter() {
                    ingest(&mut points, tile);
                }
                points
            })
            .reduce_with(|mut left, right| {
                for (key, point) in right {
                    merge_point(&mut left, key, point, aggregation, band_count);
                }
                left
            })
            .unwrap_or_default()
    };
    if points.is_empty() {
        return Ok(None);
    }

    if aggregation == Aggregation::RoughAvg {
        for point in points.values_mut() {
            for index in 0..band_count {
                point.values[index] = point.values[index]
                    .map(|value| CompactOptI32::new(Some(value / point.counts[index])));
            }
        }
    }

    // `--omit-zero` means physical zero, independently of any representative
    // value chosen by quantization. Apply it while points still carry raw
    // values so a positive value in a class that emits zero remains present.
    if apply_omits(&mut points, &product.spec.omit_zero, band_count) {
        return Ok(None);
    }

    // Quantize before the polygons are grouped, so that cells sharing a class
    // merge into a single polygon. Doing it while encoding attributes would
    // leave the geometry fragmented, which is where most of a tile's bytes are.
    // It has to run after aggregation, or averages would be taken over values
    // that were already coarsened.
    if product.spec.quantize.iter().any(Option::is_some) {
        for point in points.values_mut() {
            for (index, quantize) in product.spec.quantize.iter().enumerate() {
                if let Some(quantize) = quantize {
                    point.values[index] = point.values[index]
                        .map(|value| CompactOptI32::new(Some(quantize.class_of(value))));
                }
            }
        }
    }

    // `--omit-class` targets the representative values of quantized classes,
    // so it runs after raw values have become class indices.
    if apply_omits(&mut points, &product.spec.omit_class, band_count) {
        return Ok(None);
    }

    let band_scales = band_scales(product);

    match output {
        TileOutput::Mvt(layout) => mvt::render_mvt_layer(
            layer_name,
            &product.spec,
            layout,
            &tile_context.bounds,
            &points,
            &band_scales,
        ),
        TileOutput::Raster(layout) => raster::render_raster_tile(
            &product.spec,
            tile_context.zxy,
            &points,
            &band_scales,
            layout,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The chunk ranges rely on the Hilbert curve's self-similarity: a tile's
    /// range has to be exactly the four ranges of its children, with no gap and
    /// no overlap. Passing the zoom to `xy2h` in the wrong argument position
    /// still returns an id, so this pins the property rather than the numbers.
    #[test]
    fn a_tiles_chunk_range_is_partitioned_by_its_children() {
        let base_z = 4;
        for z in 0..base_z {
            for x in 0..(1u32 << z) {
                for y in 0..(1u32 << z) {
                    let (begin, end) = zxy_to_chunk_id_range(base_z, (z, x, y));
                    let mut children = [
                        (2 * x, 2 * y),
                        (2 * x + 1, 2 * y),
                        (2 * x, 2 * y + 1),
                        (2 * x + 1, 2 * y + 1),
                    ]
                    .map(|(cx, cy)| zxy_to_chunk_id_range(base_z, (z + 1, cx, cy)));
                    children.sort_unstable();

                    assert_eq!(children[0].0, begin, "z={z} ({x},{y})");
                    assert_eq!(children[3].1, end, "z={z} ({x},{y})");
                    for pair in children.windows(2) {
                        assert_eq!(pair[0].1, pair[1].0, "gap or overlap at z={z} ({x},{y})");
                    }
                }
            }
        }
    }

    /// At the base zoom every tile takes exactly one chunk, and together they
    /// cover the whole range without repeating one.
    #[test]
    fn base_zoom_tiles_map_one_to_one_onto_the_chunk_ids() {
        let base_z = 4;
        let mut seen = vec![false; 1 << (2 * base_z)];
        for x in 0..(1u32 << base_z) {
            for y in 0..(1u32 << base_z) {
                let (begin, end) = zxy_to_chunk_id_range(base_z, (base_z, x, y));
                assert_eq!(end, begin + 1, "({x},{y})");
                assert!(!seen[begin as usize], "chunk {begin} claimed twice");
                seen[begin as usize] = true;
            }
        }
        assert!(seen.into_iter().all(|hit| hit));
    }

    /// A product of dense cells across several chunks, some 1 km among the
    /// 250 m ones, with values a weighted statistic can tell apart.
    fn mixed_product() -> PreparedProduct {
        use gpv_products::{
            model::{BandSpec, LngLatGrid},
            products::{GpvProductElement, GpvProductIdentifier, PointValue, ProductData},
        };
        let mut points = Vec::new();
        let mut state = 12345u64;
        let mut random = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as i32
        };
        // 250 m cells, with gaps, then 1 km cells in a band of their own.
        for y in 2000..2600u16 {
            for x in 3000..3700u16 {
                if random() % 4 != 0 {
                    points.push((x, y, 0, random() % 50 - 10));
                }
            }
        }
        for y in (2600..2800u16).step_by(4) {
            for x in (3000..3700u16).step_by(4) {
                if random() % 3 != 0 {
                    points.push((x, y, 2, random() % 50 - 10));
                }
            }
        }
        let product_data = ProductData {
            points: points
                .into_iter()
                .map(|(x, y, point_power, value)| PointValue {
                    x,
                    y,
                    point_id: 0,
                    point_power,
                    band_idx: 0,
                    value,
                })
                .collect(),
            band_specs: vec![BandSpec {
                name: "value".to_string(),
                ..Default::default()
            }],
            grid: Some(LngLatGrid {
                lng_0: 118.0,
                lat_0: 20.0,
                lng_denom: 320.0,
                lat_denom: 480.0,
            }),
        };
        let product_id = GpvProductIdentifier {
            kind: GpvProductElement::HiresNowcastIntensity,
            generating_process: gpv_products::products::GeneratingProcessType::Forecast,
            ..Default::default()
        };
        crate::prepare::prepare_product(product_id, product_data)
    }

    /// Each block reduced straight from the cells, counting each once.
    fn expected_blocks(
        product: &PreparedProduct,
        chunks: &[&[(u64, crate::model::BaseTile)]],
        scale: u8,
    ) -> foldhash::HashMap<(u32, u32), i32> {
        let mask = !((1u32 << scale) - 1);
        let mut cells = foldhash::HashMap::<(u16, u16), (u8, i32)>::default();
        for chunk in chunks {
            for (_, tile) in chunk.iter() {
                for index in 0..tile.point_positions.len() {
                    if let Some(value) = tile.bands[0].values[index].get() {
                        cells.insert(
                            tile.point_positions[index],
                            (tile.point_powers[index], value),
                        );
                    }
                }
            }
        }
        let mut blocks = foldhash::HashMap::<(u32, u32), Vec<(u8, i32)>>::default();
        for ((x, y), cell) in cells {
            blocks
                .entry((u32::from(x) & mask, u32::from(y) & mask))
                .or_default()
                .push(cell);
        }
        blocks
            .into_iter()
            .map(|(key, cells)| {
                let power = cells
                    .iter()
                    .map(|&(power, _)| power)
                    .max()
                    .unwrap()
                    .max(scale);
                let mut samples = reduce::BlockSamples::default();
                for (cell_power, value) in cells {
                    samples.push(value, 1 << (2 * u32::from(cell_power)));
                }
                let value = samples
                    .reduce(
                        product.spec.reducer,
                        1 << (2 * u32::from(power)),
                        product.spec.fill[0],
                    )
                    .unwrap();
                (key, value)
            })
            .collect()
    }

    /// Cells near a chunk edge are copied into both chunks; a weighted block
    /// has to count them once however many of those chunks a tile reads.
    #[test]
    fn weighted_blocks_count_every_cell_once() {
        let mut product = mixed_product();
        let base_z = product.spec.base_z;
        for reducer in ["mean", "p50", "p95"] {
            for fill in [None, Some(0)] {
                product.spec.reducer = reducer.parse().unwrap();
                product.spec.fill = vec![fill];
                for z in [base_z - 3, base_z - 1, base_z, base_z + 2] {
                    let (lng, lat): (f64, f64) = (118.0 + 3300.0 / 320.0, 20.0 + 2300.0 / 480.0);
                    let n = f64::from(1u32 << z);
                    let x = ((lng + 180.0) / 360.0 * n) as u32;
                    let y = ((1.0
                        - (lat.to_radians().tan() + 1.0 / lat.to_radians().cos()).ln()
                            / std::f64::consts::PI)
                        / 2.0
                        * n) as u32;
                    for output in [
                        TileOutput::Mvt(MvtLayout::default()),
                        TileOutput::Raster(RasterLayout::new(ColorEncoding::Scalar24 {
                            base: 0.0,
                            interval: 1.0,
                        })),
                    ] {
                        let chunks = source_chunks(&product, (z, x, y), output);
                        let square = ChunkSquare::read_by(base_z, (z, x, y), output);
                        for scale in [1, 3] {
                            let actual =
                                weighted_blocks(&product, &chunks, square, scale, |_, _, _| true)
                                    .into_iter()
                                    .map(|(key, point)| (key, point.values[0].get().unwrap()))
                                    .collect::<foldhash::HashMap<_, _>>();
                            let expected = expected_blocks(&product, &chunks, scale);
                            assert!(!expected.is_empty());
                            assert_eq!(
                                actual, expected,
                                "{reducer} fill {fill:?} z{z} scale {scale} {output:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    fn decode_png(png: &[u8]) -> (u32, Vec<u8>) {
        let decoder = ::png::Decoder::new(std::io::Cursor::new(png));
        let mut reader = decoder.read_info().unwrap();
        let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut pixels).unwrap();
        assert_eq!(info.width, info.height);
        (info.width, pixels)
    }

    /// A 256-pixel tile one zoom deeper holds the pixels of a quarter of a
    /// 512-pixel tile, margin included, whatever the cells are merged by.
    #[test]
    fn a_small_raster_tile_is_a_quarter_of_a_large_one() {
        let mut product = mixed_product();
        let (lng, lat): (f64, f64) = (118.0 + 3300.0 / 320.0, 20.0 + 2300.0 / 480.0);
        let encoding = ColorEncoding::Scalar16 {
            min: -20.0,
            max: 60.0,
        };
        for reducer in ["max", "p95"] {
            product.spec.reducer = reducer.parse().unwrap();
            product.spec.fill = vec![(reducer != "max").then_some(0)];
            // Merged into blocks, and cell for cell.
            for z in [6u8, 9] {
                let n = f64::from(1u32 << z);
                let x = ((lng + 180.0) / 360.0 * n) as u32;
                let y = ((1.0
                    - (lat.to_radians().tan() + 1.0 / lat.to_radians().cos()).ln()
                        / std::f64::consts::PI)
                    / 2.0
                    * n) as u32;
                let large = make_layer(
                    &TileContext::new((z, x, y)),
                    &product,
                    "layer",
                    TileOutput::Raster(RasterLayout::new(encoding)),
                )
                .unwrap()
                .unwrap();
                let (large_side, large) = decode_png(&large);
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let small = make_layer(
                        &TileContext::new((z + 1, 2 * x + dx, 2 * y + dy)),
                        &product,
                        "layer",
                        TileOutput::Raster(RasterLayout {
                            encoding,
                            size: 256,
                        }),
                    )
                    .unwrap();
                    let window = |row: usize| {
                        let begin = ((row + 256 * dy as usize) * 514 + 256 * dx as usize) * 4;
                        &large[begin..begin + 258 * 4]
                    };
                    let Some(small) = small else {
                        // An empty quarter: nothing of it may be on the grid.
                        let off_grid = encoding.off_grid()[2];
                        for row in 0..258usize {
                            assert!(
                                window(row).chunks(4).all(|pixel| pixel[2] == off_grid),
                                "{reducer} z{z} quarter ({dx},{dy}) is empty but row {row} is not"
                            );
                        }
                        continue;
                    };
                    let (small_side, small) = decode_png(&small);
                    assert_eq!((large_side, small_side), (514, 258));
                    for row in 0..258usize {
                        assert_eq!(
                            &small[row * 258 * 4..(row + 1) * 258 * 4],
                            window(row),
                            "{reducer} z{z} quarter ({dx},{dy}) row {row}"
                        );
                    }
                }
            }
        }
    }

    fn value(value: i32) -> CompactOptI32 {
        CompactOptI32::new(Some(value))
    }

    #[test]
    fn omit_zero_keeps_positive_values_that_quantize_to_zero() {
        let spec = gpv_products::model::BandSpec {
            name: "value".to_string(),
            ..Default::default()
        };
        let zero_omits = crate::quantize::resolve_zero_omits(true, std::slice::from_ref(&spec));
        let quantize = crate::quantize::resolve(&["0,5".to_string()], &[spec])
            .unwrap()
            .pop()
            .flatten()
            .unwrap();
        let mut points = PointMap::from_iter([
            (
                (0, 0),
                Point {
                    values: [
                        value(0),
                        CompactOptI32::NONE,
                        CompactOptI32::NONE,
                        CompactOptI32::NONE,
                    ],
                    ..Default::default()
                },
            ),
            (
                (1, 0),
                Point {
                    values: [
                        value(1),
                        CompactOptI32::NONE,
                        CompactOptI32::NONE,
                        CompactOptI32::NONE,
                    ],
                    ..Default::default()
                },
            ),
        ]);

        assert!(!apply_omits(&mut points, &zero_omits, 1));
        assert!(!points.contains_key(&(0, 0)));
        let positive = points.get_mut(&(1, 0)).unwrap();
        positive.values[0] =
            positive.values[0].map(|raw| CompactOptI32::new(Some(quantize.class_of(raw))));

        assert_eq!(positive.values[0], value(0));
        assert_eq!(quantize.output(0), 0.0);
    }

    fn band(scaling: bool, decimal_scale: f64) -> BandScale<'static> {
        BandScale {
            name: "value",
            scaling,
            reference: 0.0,
            binary_scale: 1.0,
            decimal_scale,
            quantize: None,
        }
    }

    #[test]
    fn scaled_band_emits_double_even_when_the_value_is_integral() {
        let band = band(true, 0.1);

        // 1.0 and 1.5 must land in the same column type.
        assert_eq!(band.tag_value(10), MvtValue::Double(1.0));
        assert_eq!(band.tag_value(15), MvtValue::Double(1.5));
    }

    #[test]
    fn scaled_band_keeps_values_beyond_the_f32_integer_range() {
        let band = band(true, 1.0);

        // 2^24 is where f32 stops holding consecutive integers, and i32 would
        // saturate above 2^31.
        assert_eq!(band.tag_value(33_554_450), MvtValue::Double(33_554_450.0));
        assert_eq!(band.tag_value(i32::MAX), MvtValue::Double(2_147_483_647.0));
        assert_eq!(band.tag_value(-33_554_450), MvtValue::Double(-33_554_450.0));
    }

    #[test]
    fn a_quantized_band_emits_the_class_output_as_one_type() {
        let spec = gpv_products::model::BandSpec {
            name: "value".to_string(),
            ..Default::default()
        };
        let quantize = crate::quantize::resolve(&["0,1,2".to_string()], &[spec])
            .unwrap()
            .pop()
            .flatten()
            .unwrap();
        let band = BandScale {
            name: "value",
            scaling: true,
            reference: 0.0,
            binary_scale: 1.0,
            decimal_scale: 0.1,
            quantize: Some(&quantize),
        };

        // Integral outputs collapse the column to sint, and the value passed in
        // is a class index rather than a raw value.
        assert_eq!(band.tag_value(0), MvtValue::SInt(0));
        assert_eq!(band.tag_value(1), MvtValue::SInt(1));
        assert_eq!(band.tag_value(2), MvtValue::SInt(2));
    }

    #[test]
    fn a_quantized_band_with_fractional_outputs_stays_double() {
        let spec = gpv_products::model::BandSpec {
            name: "value".to_string(),
            ..Default::default()
        };
        let quantize = crate::quantize::resolve(&["0:0,1:0.5,2:1.5".to_string()], &[spec])
            .unwrap()
            .pop()
            .flatten()
            .unwrap();
        let band = BandScale {
            name: "value",
            scaling: true,
            reference: 0.0,
            binary_scale: 1.0,
            decimal_scale: 0.1,
            quantize: Some(&quantize),
        };

        assert_eq!(band.tag_value(0), MvtValue::Double(0.0));
        assert_eq!(band.tag_value(1), MvtValue::Double(0.5));
        assert_eq!(band.tag_value(2), MvtValue::Double(1.5));
    }

    #[test]
    fn unscaled_band_emits_sint_on_both_sides_of_zero() {
        let band = band(false, 1.0);

        assert_eq!(band.tag_value(-3), MvtValue::SInt(-3));
        assert_eq!(band.tag_value(0), MvtValue::SInt(0));
        assert_eq!(band.tag_value(7), MvtValue::SInt(7));
    }

    #[test]
    fn rough_avg_aggregates_sum_and_count_in_one_pass() {
        let values = [value(7), CompactOptI32::NONE, value(-2), value(5)];

        let (sum, count) = aggregate_band(&values, Aggregation::RoughAvg);

        assert_eq!(sum, value(10));
        assert_eq!(count, 3);
    }

    #[test]
    fn non_average_aggregations_do_not_track_counts() {
        let values = [value(6), CompactOptI32::NONE, value(3)];

        for (aggregation, expected) in [
            (Aggregation::Max, 6),
            (Aggregation::Min, 3),
            (Aggregation::BitOr, 7),
        ] {
            let (aggregated, count) = aggregate_band(&values, aggregation);
            assert_eq!(aggregated, value(expected));
            assert_eq!(count, 0);
        }
    }

    #[test]
    fn merge_only_updates_active_bands() {
        let key = (1, 2);
        let mut points = PointMap::default();
        points.insert(
            key,
            Point {
                values: [value(10), value(20), value(30), value(40)],
                counts: [1, 2, 3, 4],
                power: 1,
            },
        );

        merge_point(
            &mut points,
            key,
            Point {
                values: [value(5), value(7), value(11), value(13)],
                counts: [5, 6, 7, 8],
                power: 2,
            },
            Aggregation::RoughAvg,
            2,
        );

        let point = &points[&key];
        assert_eq!(point.values, [value(15), value(27), value(30), value(40)]);
        assert_eq!(point.counts, [6, 8, 3, 4]);
        assert_eq!(point.power, 2);
    }

    #[test]
    fn non_average_merge_does_not_update_counts_or_trailing_bands() {
        let key = (1, 2);

        for (aggregation, expected) in [
            (Aggregation::Max, [value(10), value(20)]),
            (Aggregation::Min, [value(5), value(7)]),
            (Aggregation::BitOr, [value(15), value(23)]),
        ] {
            let mut points = PointMap::default();
            points.insert(
                key,
                Point {
                    values: [value(10), value(20), value(30), value(40)],
                    counts: [1, 2, 3, 4],
                    power: 1,
                },
            );

            merge_point(
                &mut points,
                key,
                Point {
                    values: [value(5), value(7), value(11), value(13)],
                    counts: [5, 6, 7, 8],
                    power: 2,
                },
                aggregation,
                2,
            );

            let point = &points[&key];
            assert_eq!(&point.values[..2], &expected);
            assert_eq!(&point.values[2..], &[value(30), value(40)]);
            assert_eq!(point.counts, [1, 2, 3, 4]);
            assert_eq!(point.power, 2);
        }
    }
}
