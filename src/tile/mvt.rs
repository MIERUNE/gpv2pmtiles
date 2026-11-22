use anyhow::Result;
use fast_mvt::{MvtGeometry, MvtLayerBuilder, MvtMultiPolygon};
use foldhash::HashMap;
use rayon::prelude::*;

use crate::{
    geo::lat_to_web_mercator_y,
    model::{CompactOptI32, TilesetSpec},
    tile::{
        BandScale, MvtLayout, Point, PointMap, TileBounds,
        rect_union::{IntRect, union_rectangles},
    },
};

pub(super) fn render_mvt_layer(
    layer_name: &str,
    tileset_spec: &TilesetSpec,
    layout: MvtLayout,
    tile_bounds: &TileBounds,
    deduped_points: &PointMap<(u32, u32), Point>,
    band_scales: &Vec<BandScale<'_>>,
) -> Result<Option<Vec<u8>>> {
    let grouped_rectangles = make_rectangles_grouped_by_value(
        deduped_points,
        &tileset_spec.grid_spec,
        tile_bounds,
        layout,
    );
    encode_layer(
        layer_name,
        tileset_spec,
        layout.extent,
        grouped_rectangles,
        band_scales,
    )
}

/// Renders a layer from a lattice of `2^detail` pixels across the tile, each
/// one unit of an extent of the same size.
///
/// The pixels are gathered as a raster tile's are, so the blocks line up with
/// the tile instead of the source grid: none straddles an edge, and none is so
/// small that rounding to the extent loses it. The margin is whole pixels of
/// the same lattice, so that an outline drawn along a polygon falls outside
/// the tile rather than along its edge.
#[allow(clippy::too_many_arguments)]
pub(super) fn render_lattice_layer(
    layer_name: &str,
    product: &crate::model::PreparedProduct,
    zxy: super::Zxy,
    chunks: &[&[(u64, crate::model::BaseTile)]],
    layout: MvtLayout,
    level: u8,
    band_scales: &Vec<BandScale<'_>>,
    work: &mut super::reduce::Workspace,
) -> Result<Option<Vec<u8>>> {
    let spec = &product.spec;
    let bands = spec.band_specs.len();
    let across = 1u32 << layout.detail.at(zxy.0);
    let margin = layout.buffer * across / 256;
    let side = (across + 2 * margin) as usize;
    let corner = (
        i64::from(zxy.2) * i64::from(across) - i64::from(margin),
        i64::from(zxy.1) * i64::from(across) - i64::from(margin),
    );
    let finished = |b: usize, raw: Option<f64>| {
        CompactOptI32::new(
            raw.and_then(|raw| crate::quantize::postprocess(raw, b, spec))
                .map(|raw| raw.round_ties_even() as i32),
        )
    };

    // (row, column, values) of the pixels with a value, row by row.
    let mut cells = Vec::new();
    if let Some(pyramid) = super::raster::nested_pyramid(product, level, corner, side) {
        pyramid.window(level, corner, side, |row, column, raw| {
            let mut values = [CompactOptI32::NONE; 4];
            for (b, value) in values.iter_mut().enumerate().take(bands) {
                *value = finished(b, Some(raw[b]).filter(|raw| !raw.is_nan()));
            }
            if values.iter().any(|value| value.get().is_some()) {
                cells.push((row, column, values));
            }
        });
    } else {
        let Some(size) = super::raster::aggregate_pixels(
            product,
            zxy,
            chunks,
            super::raster::PixelLattice {
                across,
                margin,
                bands,
                every_band: false,
                output: super::TileOutput::Mvt(layout),
            },
            work,
        ) else {
            return Ok(None);
        };
        let super::reduce::Workspace {
            entries,
            scratch,
            pixels,
            methods,
            capacities,
            touched,
        } = work;
        touched.sort_unstable();
        for &index in touched.iter() {
            let index = index as usize;
            let mut values = [CompactOptI32::NONE; 4];
            for (b, value) in values.iter_mut().enumerate().take(bands) {
                *value = finished(
                    b,
                    pixels[index * bands + b].finish(
                        methods[index],
                        capacities[index],
                        spec.fill[b],
                        entries,
                        scratch,
                    ),
                );
            }
            if values.iter().any(|value| value.get().is_some()) {
                cells.push((index / size, index % size, values));
            }
        }
    }

    // Runs of equal values along a row become one rectangle, which leaves the
    // union a fraction of the edges to merge.
    let margin = margin as i32;
    let mut grouped_rectangles: HashMap<[CompactOptI32; 4], Vec<IntRect>> = HashMap::default();
    let mut run: Option<(usize, usize, usize, [CompactOptI32; 4])> = None;
    let mut flush = |run: Option<(usize, usize, usize, [CompactOptI32; 4])>| {
        if let Some((row, first, end, values)) = run {
            grouped_rectangles
                .entry(values)
                .or_default()
                .push(IntRect::new(
                    first as i32 - margin,
                    row as i32 - margin,
                    end as i32 - margin,
                    row as i32 + 1 - margin,
                ));
        }
    };
    for (row, column, values) in cells {
        run = match run {
            Some((run_row, first, end, run_values))
                if run_row == row && end == column && run_values == values =>
            {
                Some((row, first, column + 1, values))
            }
            other => {
                flush(other);
                Some((row, column, column + 1, values))
            }
        };
    }
    flush(run);

    encode_layer(layer_name, spec, across, grouped_rectangles, band_scales)
}

/// Merges each group of rectangles into polygons and writes them as the
/// features of one layer.
fn encode_layer(
    layer_name: &str,
    tileset_spec: &TilesetSpec,
    extent: u32,
    grouped_rectangles: HashMap<[CompactOptI32; 4], Vec<IntRect>>,
    band_scales: &Vec<BandScale<'_>>,
) -> Result<Option<Vec<u8>>> {
    if grouped_rectangles.is_empty() {
        return Ok(None);
    }

    // Encode geometries
    let value_geom = grouped_rectangles
        .into_par_iter()
        .map(|(values, rectangles)| {
            Ok((
                values,
                MvtGeometry::MultiPolygon(MvtMultiPolygon::new(union_rectangles(rectangles)?)),
            ))
        })
        .collect::<Result<Vec<_>>>()?;

    // The groups came out of a hash map, whose order is seeded afresh on every
    // run, so the features landed in the tile in a different order each time.
    // The tiles were equivalent but never byte-identical, which rules out
    // caching, delta distribution and signing. Ordering by the values makes the
    // output a function of the input alone.
    let mut value_geom = value_geom;
    value_geom.sort_unstable_by_key(|(values, _)| *values);

    // Encode features
    let mut layer = MvtLayerBuilder::new(layer_name)?;
    layer.extent(std::num::NonZeroU32::new(extent).expect("MVT extent is positive"));
    for (values, geometry) in value_geom {
        let mut feature = layer.feature(&geometry)?;
        for (band, raw_value) in band_scales.iter().zip(values) {
            let Some(value) = raw_value.get() else {
                continue;
            };
            feature.tag(band.name, band.tag_value(value))?;
        }

        // An id is only set when it can be made unique. Casting a negative
        // first band straight to `u64` sign-extended it, filling the high half
        // with ones and swallowing whatever the second band was OR-ed into; the
        // three- and four-band forms overlapped their bit ranges outright, so
        // features with different values shared an id. Two bands are the most
        // that fit exactly, and beyond that no id is better than a colliding
        // one - the specification asks for uniqueness, and consumers use it to
        // carry per-feature state.
        if let Some(feature_id) = unique_feature_id(&values[..tileset_spec.band_specs.len()]) {
            feature.id(Some(feature_id));
        }
        layer = feature.end();
    }

    Ok(Some(layer.encode()))
}

/// A feature id that no other value combination in the layer can share.
///
/// Each band is a whole `i32`, so only one or two of them fit in the 64 bits an
/// id has. A band without a value has no code left to stand for it, so those
/// features go without an id rather than share one.
fn unique_feature_id(values: &[CompactOptI32]) -> Option<u64> {
    let present = |value: &CompactOptI32| value.get().map(|value| u64::from(value as u32));
    match values {
        [first] => present(first),
        [first, second] => Some((present(first)? << 32) | present(second)?),
        _ => None,
    }
}

/// Creates polygons from data points
fn make_rectangles_grouped_by_value(
    deduped_points: &PointMap<(u32, u32), Point>,
    grid_spec: &gpv_products::model::LngLatGrid,
    tile_bounds: &TileBounds,
    layout: MvtLayout,
) -> HashMap<[CompactOptI32; 4], Vec<IntRect>> {
    let mut grouped_rectangles: HashMap<[CompactOptI32; 4], Vec<IntRect>> = HashMap::default();
    // Both are bounded by `validate_options`, so neither overflows an i32.
    let extent = layout.extent as i32;
    let buffer = (layout.buffer * layout.extent / 256) as i32;
    let tile_width = tile_bounds.mx2 - tile_bounds.mx1;
    let w = (extent as f64) / tile_width;
    let mut projected_y = HashMap::<(u32, u32), (f64, f64)>::default();

    for ((x, y), point) in deduped_points {
        let value = point.values;
        let width = 1 << point.power;
        let lng1 = grid_spec.lng_0 + (*x as f64 - 0.5) / grid_spec.lng_denom as f64;
        let lng2 = grid_spec.lng_0 + ((*x + width) as f64 - 0.5) / grid_spec.lng_denom as f64;
        let mx1 = (lng1 + 180.0) / 360.0;
        let mx2 = (lng2 + 180.0) / 360.0;
        let (my1, my2) = *projected_y.entry((*y, width)).or_insert_with(|| {
            let lat2 = grid_spec.lat_0 + (*y as f64 - 0.5) / grid_spec.lat_denom as f64;
            let lat1 = grid_spec.lat_0 + ((*y + width) as f64 - 0.5) / grid_spec.lat_denom as f64;
            (lat_to_web_mercator_y(lat1), lat_to_web_mercator_y(lat2))
        });
        let (mx1, mx2) = if mx2 > 1. {
            (mx1 - 1., mx2 - 1.)
        } else {
            (mx1, mx2)
        };

        let tx1 =
            (((mx1 - tile_bounds.mx1) * w + 0.5).floor() as i32).clamp(-buffer, extent + buffer);
        let tx2 =
            (((mx2 - tile_bounds.mx1) * w + 0.5).floor() as i32).clamp(-buffer, extent + buffer);
        let ty1 =
            (((my1 - tile_bounds.my1) * w + 0.5).floor() as i32).clamp(-buffer, extent + buffer);
        let ty2 =
            (((my2 - tile_bounds.my1) * w + 0.5).floor() as i32).clamp(-buffer, extent + buffer);

        if ty1 < ty2 {
            if tx1 < tx2 {
                grouped_rectangles
                    .entry(value)
                    .or_default()
                    .push(IntRect::new(tx1, ty1, tx2, ty2));
            }
            // If wrap around anti-meridian
            // TODO: optimization?
            if mx1 < 0. && mx2 > 0. {
                let tx1 = (((mx1 + 1. - tile_bounds.mx1) / tile_width * (extent as f64) + 0.5)
                    .floor() as i32)
                    .clamp(-buffer, extent + buffer);
                let tx2 = (((mx2 + 1. - tile_bounds.mx1) / tile_width * (extent as f64) + 0.5)
                    .floor() as i32)
                    .clamp(-buffer, extent + buffer);
                if tx1 < tx2 {
                    grouped_rectangles
                        .entry(value)
                        .or_default()
                        .push(IntRect::new(tx1, ty1, tx2, ty2));
                }
            }
        }
    }

    grouped_rectangles
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(value: i32) -> CompactOptI32 {
        CompactOptI32::new(Some(value))
    }

    /// A negative first band used to sign-extend into the high half of the id,
    /// which both swallowed the second band and collided with every other
    /// negative value that did the same.
    #[test]
    fn a_negative_band_does_not_swallow_the_one_beside_it() {
        let left = unique_feature_id(&[value(-1), value(7)]);
        let right = unique_feature_id(&[value(-1), value(9)]);

        assert_ne!(left, right, "the second band has to reach the id");
        assert_eq!(left, Some(0xFFFF_FFFF_0000_0007));
    }

    #[test]
    fn distinct_value_pairs_get_distinct_ids() {
        let pairs = [
            [value(0), value(0)],
            [value(0), value(1)],
            [value(1), value(0)],
            [value(-1), value(0)],
            [value(0), value(-1)],
            // i32::MIN is reserved for "missing", so the extremes a band can
            // actually carry are one in from it.
            [value(i32::MIN + 1), value(i32::MAX)],
            [value(i32::MAX), value(i32::MIN + 1)],
        ];
        let ids = pairs
            .iter()
            .map(|pair| unique_feature_id(pair))
            .collect::<Vec<_>>();

        for (index, id) in ids.iter().enumerate() {
            assert!(id.is_some());
            assert!(
                !ids[index + 1..].contains(id),
                "{:?} shares an id with a later pair",
                pairs[index]
            );
        }
    }

    /// Three bands are 96 bits, which cannot be squeezed into 64 without two
    /// combinations meeting. Going without an id is the honest outcome.
    #[test]
    fn more_bands_than_fit_get_no_id() {
        assert_eq!(unique_feature_id(&[value(1), value(2), value(3)]), None);
        assert_eq!(
            unique_feature_id(&[value(1), value(2), value(3), value(4)]),
            None
        );
    }

    fn grid() -> gpv_products::model::LngLatGrid {
        // One-degree columns and 2/3-degree rows whose first cell starts at
        // 10W 10S.
        gpv_products::model::LngLatGrid {
            lng_0: -10.0 + 0.5,
            lat_0: -10.0 + 0.5 / 1.5,
            lng_denom: 1.0,
            lat_denom: 1.5,
        }
    }

    /// Tile 2/2/1, which spans 0-90E and 0-66.5N.
    const TILE: TileBounds = TileBounds {
        mx1: 0.5,
        my1: 0.25,
        mx2: 0.75,
        my2: 0.5,
    };

    fn cell(dn: i32, power: u8) -> Point {
        let mut values = [CompactOptI32::NONE; 4];
        values[0] = value(dn);
        Point {
            values,
            counts: [1, 0, 0, 0],
            power,
        }
    }

    /// One cell reaching past every edge of the tile, so the rectangle is
    /// whatever the buffer clamps it to.
    fn rectangles_past_every_edge(layout: MvtLayout) -> Vec<IntRect> {
        let mut points = PointMap::default();
        points.insert((0, 0), cell(1, 7));

        make_rectangles_grouped_by_value(&points, &grid(), &TILE, layout)
            .into_values()
            .flatten()
            .collect()
    }

    #[test]
    fn the_buffer_is_counted_in_pixels_of_a_256px_tile() {
        assert_eq!(
            rectangles_past_every_edge(MvtLayout::default()),
            [IntRect::new(-32, -32, 4128, 4128)]
        );
        assert_eq!(
            rectangles_past_every_edge(MvtLayout {
                extent: 256,
                buffer: 1,
                ..Default::default()
            }),
            [IntRect::new(-1, -1, 257, 257)]
        );
    }

    #[test]
    fn a_fixed_layer_carries_the_extent_and_an_integer_dn() {
        let spec = TilesetSpec {
            name: String::new(),
            base_z: 0,
            grid_spec: grid(),
            reducer: crate::tile::Reducer::Legacy(gpv_products::model::Aggregation::Max),
            fill: vec![None],
            band_specs: vec![gpv_products::model::BandSpec {
                name: "DN".to_string(),
                ..Default::default()
            }],
            quantize: vec![None],
            omit_zero: vec![None],
            omit_class: vec![None],
            bounds: [0.0; 4],
        };
        let scales = vec![BandScale {
            name: "DN",
            scaling: false,
            reference: 0.0,
            binary_scale: 1.0,
            decimal_scale: 1.0,
            quantize: None,
        }];
        let mut points = PointMap::default();
        points.insert((0, 0), cell(-1, 6));
        points.insert((64, 0), cell(4, 6));
        let layout = MvtLayout {
            extent: 256,
            buffer: 1,
            ..Default::default()
        };

        let bytes = render_mvt_layer("dosha", &spec, layout, &TILE, &points, &scales)
            .unwrap()
            .unwrap();

        let reader = fast_mvt::MvtReaderRef::new(&bytes).unwrap();
        let layer = reader.layers().next().unwrap();
        assert_eq!(layer.name(), "dosha");
        assert_eq!(layer.version(), 2);
        assert_eq!(layer.extent(), 256);
        assert_eq!(layer.keys(), ["DN"]);
        let features = layer
            .features()
            .map(|feature| {
                let properties = feature.properties_vec().unwrap();
                (feature.id(), properties[0].0, properties[0].1)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            features,
            [
                (Some(0xFFFF_FFFF), "DN", fast_mvt::MvtValueRef::SInt(-1)),
                (Some(4), "DN", fast_mvt::MvtValueRef::SInt(4)),
            ]
        );
    }

    /// A missing band has no code of its own, so the feature cannot be told
    /// apart from one that happens to carry that code.
    #[test]
    fn a_missing_band_leaves_the_feature_without_an_id() {
        assert_eq!(unique_feature_id(&[CompactOptI32::NONE]), None);
        assert_eq!(unique_feature_id(&[value(1), CompactOptI32::NONE]), None);
    }
}
