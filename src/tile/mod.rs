mod color_map;
mod detail;
mod mvt;
mod pyramid;
mod raster;
mod reduce;

pub(crate) use color_map::ColorMap;
pub(crate) use detail::DetailSchedule;
pub(crate) use pyramid::Pyramid;
pub(crate) use reduce::Reducer;

pub(crate) use raster::{
    ColorEncoding, RangeReport, RasterLayout, TILE_BUFFER as RASTER_TILE_BUFFER,
    TILE_SIZES as RASTER_TILE_SIZES,
};
mod rect_union;

use anyhow::Result;
use fast_mvt::MvtValue;

use fast_hilbert::xy2h;

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
fn aggregate_band(values: &[CompactOptI32], aggregation: Reducer) -> CompactOptI32 {
    // This path only deduplicates copies of one source cell. A mean or
    // percentile has the source value itself; strict preparation checked it.
    CompactOptI32::new(match aggregation {
        Reducer::Max | Reducer::Mean | Reducer::Percentile(_) => values.iter().flatten().max(),
        Reducer::Min => values.iter().flatten().min(),
        Reducer::BitOr => values.iter().flatten().reduce(|left, right| left | right),
    })
}

fn merge_point(
    points: &mut PointMap<(u32, u32), Point>,
    key: (u32, u32),
    incoming: Point,
    aggregation: Reducer,
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
                Reducer::Max | Reducer::Mean | Reducer::Percentile(_) => {
                    for index in 0..band_count {
                        if let Some(value) = incoming.values[index].get() {
                            point.values[index] = CompactOptI32::new(Some(
                                point.values[index].unwrap_or(value).max(value),
                            ));
                        }
                    }
                }
                Reducer::Min => {
                    for index in 0..band_count {
                        if let Some(value) = incoming.values[index].get() {
                            point.values[index] = CompactOptI32::new(Some(
                                point.values[index].unwrap_or(value).min(value),
                            ));
                        }
                    }
                }
                Reducer::BitOr => {
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

fn owns_cell(
    square: ChunkSquare,
    (column, row): (u32, u32),
    tile: &crate::model::BaseTile,
    index: usize,
) -> bool {
    let (dx, dy) = tile.point_homes[index];
    if (dx, dy) == (0, 0) {
        return true;
    }
    let home = (
        (i64::from(column) + i64::from(dx)).rem_euclid(i64::from(square.across)) as u32,
        (i64::from(row) + i64::from(dy)) as u32,
    );
    square.nearest(home) == (column, row)
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

/// The level `zoom + detail` a zoom aggregates its pixels at, or `None` for a
/// zoom that draws source cells instead.
///
/// MVT takes the lattice at low zooms; a raster tile wherever a cell is finer than a
/// pixel. A level is the number of pixels across the world as a power of two,
/// so zooms whose details differ can share one.
pub(crate) fn lattice_level(product: &PreparedProduct, zoom: u8, output: TileOutput) -> Option<u8> {
    let grid = &product.spec.grid_spec;
    match output {
        TileOutput::Mvt(layout) => {
            let maximum_detail_zoom = (grid.lat_denom * 360.0 * 2.0 / 512.0).log2().round() as u8;
            let detail = layout.detail.at(zoom);
            let lattice = detail::uses_lattice(
                maximum_detail_zoom,
                zoom,
                detail,
                (grid.lng_denom, grid.lat_denom),
                layout.extent,
            );
            lattice.then_some(zoom + detail)
        }
        TileOutput::Raster(layout) => {
            raster::uses_pixel_aggregation(&product.spec, (zoom, 0, 0), layout.size)
                .then_some(zoom + layout.size.trailing_zeros() as u8)
        }
    }
}

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
    if let Some(level) = lattice_level(product, z, output) {
        return reduce::with_workspace(|work| match output {
            TileOutput::Mvt(layout) => mvt::render_lattice_layer(
                layer_name,
                product,
                tile_context.zxy,
                &chunks,
                layout,
                level,
                &band_scales(product),
                work,
            ),
            TileOutput::Raster(layout) => raster::render_pixel_tile(
                product,
                tile_context.zxy,
                &chunks,
                &band_scales(product),
                layout,
                level,
                work,
            ),
        });
    }

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
    let aggregation = product.spec.reducer;
    let band_count = product.spec.band_specs.len();
    let grid = &product.spec.grid_spec;

    let ingest = |points: &mut PointMap<(u32, u32), Point>, tile: &crate::model::BaseTile| {
        debug_assert_eq!(tile.point_positions.len(), tile.point_powers.len());
        let group_key = |index: usize| {
            let (x, y) = tile.point_positions[index];
            (u32::from(x), u32::from(y), tile.point_powers[index])
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
            let power = point_power;
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
            for (band_index, band) in tile.bands.iter().enumerate() {
                values[band_index] = aggregate_band(&band.values[group_begin..end], aggregation);
            }

            merge_point(
                points,
                (grid_x, grid_y),
                Point { values, power },
                aggregation,
                band_count,
            );
        }
    };

    let mut points = {
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
    };
    if points.is_empty() {
        return Ok(None);
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
        crate::prepare::prepare_product(product_id, product_data, true).unwrap()
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

    /// Where cells are wider than a pixel but shorter than one, a strip of
    /// heavy rain one cell high has to show in every column it crosses.
    /// Sampling at pixel centres skipped the rows no centre fell in.
    #[test]
    fn a_strip_shorter_than_a_pixel_survives_at_the_zoom_cells_turn_wider() {
        use gpv_products::{
            model::{BandSpec, LngLatGrid},
            products::{
                GeneratingProcessType, GpvProductElement, GpvProductIdentifier, PointValue,
                ProductData,
            },
        };
        let grid = LngLatGrid {
            lng_0: 118.0,
            lat_0: 20.0,
            lng_denom: 320.0,
            lat_denom: 480.0,
        };
        let strip = 2200u16;
        let mut points = Vec::new();
        for y in 2000..2400u16 {
            for x in 3000..3400u16 {
                points.push(PointValue {
                    x,
                    y,
                    point_id: 0,
                    point_power: 0,
                    band_idx: 0,
                    value: if y == strip { 50 } else { 1 },
                });
            }
        }
        let product = crate::prepare::prepare_product(
            GpvProductIdentifier {
                kind: GpvProductElement::HiresNowcastIntensity,
                generating_process: GeneratingProcessType::Forecast,
                ..Default::default()
            },
            ProductData {
                points,
                band_specs: vec![BandSpec {
                    name: "value".to_string(),
                    ..Default::default()
                }],
                grid: Some(grid.clone()),
            },
            false,
        )
        .unwrap();

        // At zoom 8 a cell is about 1.14 pixels wide and 0.84 high here.
        let z = 8u8;
        let layout = RasterLayout::new(ColorEncoding::Scalar16 {
            min: 0.0,
            max: 100.0,
        });
        assert!(raster::uses_pixel_aggregation(
            &product.spec,
            (z, 0, 0),
            layout.size
        ));
        let n = f64::from(1u32 << z);
        let (lng, lat): (f64, f64) = (118.0 + 3200.0 / 320.0, 20.0 + f64::from(strip) / 480.0);
        let x = ((lng + 180.0) / 360.0 * n) as u32;
        let y = ((1.0
            - (lat.to_radians().tan() + 1.0 / lat.to_radians().cos()).ln() / std::f64::consts::PI)
            / 2.0
            * n) as u32;
        let png = make_layer(
            &TileContext::new((z, x, y)),
            &product,
            "layer",
            TileOutput::Raster(layout),
        )
        .unwrap()
        .unwrap();
        let (side, pixels) = decode_png(&png);
        let side = side as usize;
        let code = |row: usize, column: usize| {
            let pixel = &pixels[(row * side + column) * 4..(row * side + column) * 4 + 4];
            (pixel[2] == 255).then_some(u16::from(pixel[0]) << 8 | u16::from(pixel[1]))
        };
        let mut counts = std::collections::HashMap::<u16, usize>::new();
        for row in 0..side {
            for column in 0..side {
                if let Some(code) = code(row, column) {
                    *counts.entry(code).or_default() += 1;
                }
            }
        }
        let background = counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .unwrap()
            .0;
        let mut columns_on_grid = 0;
        for column in 0..side {
            if (0..side).all(|row| code(row, column).is_none()) {
                continue;
            }
            columns_on_grid += 1;
            assert!(
                (0..side).any(|row| code(row, column).is_some_and(|code| code > background)),
                "the strip is missing from column {column}"
            );
        }
        assert!(columns_on_grid > 100);
    }

    /// A lattice layer is one unit square per pixel with a value, margin
    /// included, merged: its polygons cover exactly as many units as pixels
    /// hold a value.
    #[test]
    fn a_lattice_layer_covers_one_unit_per_pixel_with_a_value() {
        let mut product = mixed_product();
        product.spec.reducer = "p95".parse().unwrap();
        product.spec.fill = vec![Some(0)];
        let (lng, lat): (f64, f64) = (118.0 + 3300.0 / 320.0, 20.0 + 2300.0 / 480.0);
        for (detail, z) in [8u8, 7]
            .into_iter()
            .flat_map(|d| [4u8, 6, 7].map(|z| (d, z)))
        {
            let across = 1u32 << detail;
            let layout = MvtLayout {
                detail: detail.to_string().parse().unwrap(),
                ..Default::default()
            };
            let n = f64::from(1u32 << z);
            let zxy = (
                z,
                ((lng + 180.0) / 360.0 * n) as u32,
                ((1.0
                    - (lat.to_radians().tan() + 1.0 / lat.to_radians().cos()).ln()
                        / std::f64::consts::PI)
                    / 2.0
                    * n) as u32,
            );
            let bytes = make_layer(
                &TileContext::new(zxy),
                &product,
                "layer",
                TileOutput::Mvt(layout),
            )
            .unwrap()
            .unwrap();
            let reader = fast_mvt::MvtReaderRef::new(&bytes).unwrap();
            let layer = reader.layers().next().unwrap();
            assert_eq!(layer.extent(), across, "detail {detail} z{z}");
            let ring_area = |ring: &fast_mvt::MvtLineString| {
                ring.0
                    .windows(2)
                    .map(|pair| {
                        i64::from(pair[0].x) * i64::from(pair[1].y)
                            - i64::from(pair[1].x) * i64::from(pair[0].y)
                    })
                    .sum::<i64>()
                    .abs()
                    / 2
            };
            let mut area = 0;
            for feature in layer.features() {
                let polygons = match feature.geometry().unwrap() {
                    fast_mvt::MvtGeometry::MultiPolygon(polygons) => polygons.0,
                    fast_mvt::MvtGeometry::Polygon(polygon) => vec![polygon],
                    other => panic!("a lattice feature is a polygon, not {other:?}"),
                };
                for polygon in polygons {
                    area += ring_area(polygon.exterior());
                    area -= polygon.interiors().iter().map(ring_area).sum::<i64>();
                }
            }

            let chunks = source_chunks(&product, zxy, TileOutput::Mvt(layout));
            let pixels = reduce::with_workspace(|work| {
                raster::aggregate_pixels(
                    &product,
                    zxy,
                    &chunks,
                    raster::PixelLattice {
                        across,
                        margin: layout.buffer * across / 256,
                        bands: 1,
                        every_band: false,
                        output: TileOutput::Mvt(layout),
                    },
                    work,
                )
                .map_or(0, |_| work.touched.len())
            });
            assert!(pixels > 0, "detail {detail} z{z}");
            assert_eq!(area, pixels as i64, "detail {detail} z{z}");
        }
    }

    /// The pyramid stands in for gathering the cells, so every tile has to
    /// come out the same byte for byte with it as without it.
    ///
    /// A vector tile gathers only the chunks under it, so its margin comes out
    /// whole only from the pyramid; it is compared without one. A raster tile
    /// reads its neighbours, and is compared margin and all.
    #[test]
    fn the_pyramid_draws_the_tiles_the_cells_do() {
        let mut compared = 0;
        for (reducer, fill) in [
            ("max", None),
            ("max", Some(40)),
            ("min", Some(3)),
            ("mean", None),
            ("mean", Some(0)),
            ("avg", None),
            ("bit-or", Some(64)),
            ("p95", None),
            ("p95", Some(0)),
            ("p50", Some(7)),
            ("p99.5", None),
            ("p0", Some(3)),
            ("p100", None),
        ] {
            compared += pyramid_matches_cells(reducer, fill);
        }
        assert!(compared > 100, "{compared}");
    }

    /// Draws tiles round one spot with the pyramid and without, and returns
    /// how many held anything.
    fn pyramid_matches_cells(reducer: &str, fill: Option<i32>) -> usize {
        let mut product = mixed_product();
        product.spec.reducer = reducer.parse().unwrap();
        product.spec.fill = vec![fill; product.spec.band_specs.len()];
        let (lng, lat): (f64, f64) = (118.0 + 3300.0 / 320.0, 20.0 + 2300.0 / 480.0);
        let outputs = [
            TileOutput::Mvt(MvtLayout {
                buffer: 0,
                ..Default::default()
            }),
            TileOutput::Mvt(MvtLayout {
                buffer: 0,
                detail: "7".parse().unwrap(),
                ..Default::default()
            }),
            TileOutput::Raster(RasterLayout::new(ColorEncoding::Scalar16 {
                min: -20.0,
                max: 60.0,
            })),
            TileOutput::Raster(RasterLayout {
                encoding: ColorEncoding::Scalar16 {
                    min: -20.0,
                    max: 60.0,
                },
                size: 256,
            }),
        ];
        let mut compared = 0;
        for output in outputs {
            product.pyramid = None;
            crate::archive::build_pyramid(&mut product, 0, 10, output);
            assert!(product.pyramid.is_some(), "{output:?}");
            for z in 2u8..=9 {
                let n = f64::from(1u32 << z);
                let x = ((lng + 180.0) / 360.0 * n) as u32;
                let y = ((1.0
                    - (lat.to_radians().tan() + 1.0 / lat.to_radians().cos()).ln()
                        / std::f64::consts::PI)
                    / 2.0
                    * n) as u32;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let context = TileContext::new((z, x + dx, y + dy));
                    let with = make_layer(&context, &product, "layer", output).unwrap();
                    let pyramid = product.pyramid.take();
                    let without = make_layer(&context, &product, "layer", output).unwrap();
                    product.pyramid = pyramid;
                    assert_eq!(
                        with, without,
                        "{reducer} {fill:?} {output:?} z{z} +({dx},{dy})"
                    );
                    compared += usize::from(with.is_some());
                }
            }
        }
        compared
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
    fn source_cell_values_skip_missing_samples() {
        let values = [value(6), CompactOptI32::NONE, value(3)];

        for (aggregation, expected) in [(Reducer::Max, 6), (Reducer::Min, 3), (Reducer::BitOr, 7)] {
            let aggregated = aggregate_band(&values, aggregation);
            assert_eq!(aggregated, value(expected));
        }
    }

    #[test]
    fn source_cell_merge_preserves_trailing_bands() {
        let key = (1, 2);

        for (aggregation, expected) in [
            (Reducer::Max, [value(10), value(20)]),
            (Reducer::Min, [value(5), value(7)]),
            (Reducer::BitOr, [value(15), value(23)]),
        ] {
            let mut points = PointMap::default();
            points.insert(
                key,
                Point {
                    values: [value(10), value(20), value(30), value(40)],
                    power: 1,
                },
            );

            merge_point(
                &mut points,
                key,
                Point {
                    values: [value(5), value(7), value(11), value(13)],
                    power: 2,
                },
                aggregation,
                2,
            );

            let point = &points[&key];
            assert_eq!(&point.values[..2], &expected);
            assert_eq!(&point.values[2..], &[value(30), value(40)]);
            assert_eq!(point.power, 2);
        }
    }
}
