use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use anyhow::{Context, Result, ensure};
use foldhash::HashMap;
use gpv_products::{
    grib::GridSquareMessageReader,
    products::{GpvProductIdentifier, ProductData},
};
use rayon::prelude::*;
use tinygrib2::MessageReader;

use fast_hilbert::xy2h;

use crate::{
    geo::lat_to_web_mercator_y,
    model::{Band, BaseTile, CompactOptI32, PreparedProduct, TilesetSpec},
    tile::Reducer,
};

#[derive(Clone, Debug, Default)]
struct TilePoint {
    x: u16,
    y: u16,
    point_power: u8,
    home: (i8, i8),
    values: [CompactOptI32; 4],
}

#[derive(Debug)]
pub(crate) struct RawProduct {
    pub product_id: GpvProductIdentifier,
    pub data: ProductData,
}

pub(crate) fn read_products(
    input: &Path,
    retained_product: Option<&str>,
) -> Result<Vec<RawProduct>> {
    let file =
        File::open(input).with_context(|| format!("failed to open input {}", input.display()))?;
    let message_reader = if input.extension().is_some_and(|ext| ext == "gz") {
        read_messages(
            BufReader::new(flate2::read::GzDecoder::new(BufReader::new(file))),
            retained_product,
        )?
    } else {
        read_messages(BufReader::new(file), retained_product)?
    };

    Ok(message_reader
        .products
        .into_iter()
        .map(|(product_id, data)| RawProduct { product_id, data })
        .collect())
}

/// Prepares every product for tiling.
///
/// `strict` refuses source cells that overlap, sit off their own size, or
/// disagree at one origin. A weighted statistic counts each cell's area, so
/// such input would silently be counted twice; a maximum or a minimum gives the
/// same answer either way, and input that converted before keeps converting.
pub(crate) fn prepare_products(
    products: Vec<RawProduct>,
    strict: bool,
) -> Result<Vec<PreparedProduct>> {
    products
        .into_par_iter()
        .map(|product| prepare_product(product.product_id, product.data, strict))
        .collect()
}

fn read_messages<R: Read>(
    mut reader: R,
    retained_product: Option<&str>,
) -> Result<GridSquareMessageReader> {
    let mut message_reader = GridSquareMessageReader::retaining_product(retained_product);
    while let Some(()) = message_reader.read_next_message(&mut reader)? {}
    Ok(message_reader)
}

pub(crate) fn prepare_product(
    product_id: GpvProductIdentifier,
    mut product_data: ProductData,
    strict: bool,
) -> Result<PreparedProduct> {
    let grid = product_data
        .grid
        .clone()
        .unwrap_or_else(|| product_id.grid().clone());
    for point in &product_data.points {
        ensure!(point.point_power <= 16, "source cell power exceeds 16");
        let width = 1u32 << point.point_power;
        ensure!(
            !strict || (u32::from(point.x) % width == 0 && u32::from(point.y) % width == 0),
            "source cell origin is not aligned to its size"
        );
        ensure!(
            usize::from(point.band_idx) < product_data.band_specs.len(),
            "source band index out of range"
        );
    }
    // Two stable counting passes: x, then y. Bands at one origin stay together.
    for coordinate in [false, true] {
        let mut offsets = vec![0usize; 65536];
        for point in &product_data.points {
            offsets[usize::from(if coordinate { point.y } else { point.x })] += 1;
        }
        let mut sum = 0;
        for count in &mut offsets {
            let next = sum + *count;
            *count = sum;
            sum = next;
        }
        let mut sorted = (0..product_data.points.len())
            .map(|_| gpv_products::products::PointValue {
                x: 0,
                y: 0,
                point_id: 0,
                point_power: 0,
                band_idx: 0,
                value: 0,
            })
            .collect::<Vec<_>>();
        for mut point in product_data.points.drain(..) {
            let key = usize::from(if coordinate { point.y } else { point.x });
            point.point_id = (u32::from(point.y) << 16) | u32::from(point.x);
            sorted[offsets[key]] = point;
            offsets[key] += 1;
        }
        product_data.points = sorted;
    }

    let mut min_lng = f64::MAX;
    let mut max_lng = f64::MIN;
    let mut min_lat = f64::MAX;
    let mut max_lat = f64::MIN;
    let base_z = (grid.lat_denom * 360. * 2. / 512.).log2().ceil() as u8;
    let buffer = (2.0 / 256.0) / (1 << base_z) as f64;
    let mut tiles: HashMap<(u32, u32), Vec<TilePoint>> = HashMap::default();

    // Every cell edge is a grid coordinate offset by half a cell, so the
    // projections are shared: each column and row of the grid needs one
    // longitude and one Mercator latitude, not one per point. The expressions
    // are the same ones the per-point code used, in the same order, so the
    // coordinates are unchanged bit for bit.
    let max_edge = product_data
        .points
        .iter()
        .map(|point| {
            let edge = point.x as usize + (1 << point.point_power);
            edge.max(point.y as usize + (1 << point.point_power))
        })
        .max()
        .unwrap_or(0);
    let lng_edge = (0..=max_edge)
        .map(|edge| grid.lng_0 + (edge as f64 - 0.5) / grid.lng_denom as f64)
        .collect::<Vec<_>>();
    let mx_edge = lng_edge
        .iter()
        .map(|lng| (lng + 180.0) / 360.0)
        .collect::<Vec<_>>();
    let lat_edge = (0..=max_edge)
        .map(|edge| grid.lat_0 + (edge as f64 - 0.5) / grid.lat_denom as f64)
        .collect::<Vec<_>>();
    let my_edge = lat_edge
        .iter()
        .map(|&lat| lat_to_web_mercator_y(lat))
        .collect::<Vec<_>>();

    // Sparse 64x64 occupancy bitmaps. Allocate only touched squares, and only
    // for strict validation. Origin groups insert their footprint once.
    let mut occupied = HashMap::<(u32, u32), Box<[u64; 64]>>::default();
    for point_bands in product_data
        .points
        .chunk_by(|left, right| left.point_id == right.point_id)
    {
        let point = point_bands.first().expect("point group is not empty");
        if strict {
            let width = 1u32 << point.point_power;
            let x0 = u32::from(point.x);
            let x1 = x0 + width;
            let y0 = u32::from(point.y);
            for y in y0..y0 + width {
                for bx in x0 / 64..=(x1 - 1) / 64 {
                    let left = x0.saturating_sub(bx * 64);
                    let right = (x1 - bx * 64).min(64);
                    let mask = (u64::MAX << left) & (u64::MAX >> (64 - right));
                    let rows = occupied
                        .entry((bx, y / 64))
                        .or_insert_with(|| Box::new([0; 64]));
                    let row = &mut rows[(y % 64) as usize];
                    ensure!(
                        *row & mask == 0,
                        "overlapping source cells at ({}, {})",
                        point.x,
                        point.y
                    );
                    *row |= mask;
                }
            }
        }
        let mut band_values = [None; 4];
        for band in point_bands {
            ensure!(
                !strict || band.point_power == point.point_power,
                "different source cell sizes at the same origin"
            );
            let previous = &mut band_values[usize::from(band.band_idx)];
            ensure!(
                !strict || previous.is_none_or(|value| value == band.value),
                "conflicting source values at the same origin"
            );
            *previous = Some(band.value);
        }
        let width = 1 << point.point_power;
        let lng1 = lng_edge[point.x as usize];
        let lng2 = lng_edge[point.x as usize + width];
        let lat1 = lat_edge[point.y as usize + width];
        let lat2 = lat_edge[point.y as usize];

        min_lng = min_lng.min(lng1);
        max_lng = max_lng.max(lng2);
        min_lat = min_lat.min(lat2);
        max_lat = max_lat.max(lat1);

        let mx1 = mx_edge[point.x as usize];
        let mx2 = mx_edge[point.x as usize + width];
        let my1 = my_edge[point.y as usize + width];
        let my2 = my_edge[point.y as usize];
        if my1.is_nan() || my2.is_nan() {
            continue;
        }

        let x1 = ((mx1 - buffer) * (1 << base_z) as f64).floor() as i32;
        let x2 = ((mx2 + buffer) * (1 << base_z) as f64).ceil() as i32 - 1;
        let y1 = ((my1 - buffer) * (1 << base_z) as f64).floor() as i32;
        let y2 = ((my2 + buffer) * (1 << base_z) as f64).ceil() as i32 - 1;

        // A band the cell has no value for stays missing. Left at the
        // default it read as a stored zero, which a statistic then counted.
        let mut tile_point = TilePoint {
            x: point.x,
            y: point.y,
            point_power: point.point_power,
            values: [CompactOptI32::NONE; 4],
            ..Default::default()
        };
        for point_band in point_bands {
            tile_point.values[point_band.band_idx as usize] =
                CompactOptI32::new(Some(point_band.value));
        }

        // The chunk holding the cell's centre is the one whose copy counts
        // when a tile reads several; see `BaseTile::point_homes`. The centre
        // lies inside the margin the copies cover, so it is one of them.
        let chunks_across = 1i64 << base_z;
        let home_x = (((mx1 + mx2) / 2.0) * chunks_across as f64).floor() as i64;
        let home_y = ((((my1 + my2) / 2.0) * chunks_across as f64).floor() as i64)
            .clamp(0, chunks_across - 1);
        for x in x1..=x2 {
            let offset_x = (home_x - i64::from(x)).rem_euclid(chunks_across);
            // The shorter way round, so a cell on the antimeridian points
            // across it rather than all the way back.
            let offset_x = if offset_x > chunks_across / 2 {
                offset_x - chunks_across
            } else {
                offset_x
            };
            let x = x.rem_euclid(1 << base_z);
            for y in y1..=y2 {
                if !(0..1 << base_z).contains(&y) {
                    continue;
                }
                tile_point.home = (
                    i8::try_from(offset_x).expect("a cell spans only a few chunks"),
                    i8::try_from(home_y - i64::from(y)).expect("a cell spans only a few chunks"),
                );
                tiles
                    .entry((x as u32, y as u32))
                    .or_default()
                    .push(tile_point.clone());
            }
        }
    }

    let band_count = product_id.bands().len();
    assert!((1..=4).contains(&band_count));
    let mut chunks = tiles
        .into_par_iter()
        .map(|((tile_x, tile_y), points)| {
            let bands = (0..band_count)
                .map(|band_index| Band {
                    values: points
                        .iter()
                        .map(|point| point.values[band_index])
                        .collect(),
                })
                .collect();
            let tile = BaseTile {
                point_positions: points.iter().map(|point| (point.x, point.y)).collect(),
                point_powers: points.iter().map(|point| point.point_power).collect(),
                point_homes: points.iter().map(|point| point.home).collect(),
                bands,
            };
            (xy2h(tile_x, tile_y, base_z), tile)
        })
        .collect::<Vec<_>>();
    chunks.par_sort_unstable_by_key(|(tile_id, _)| *tile_id);

    let spec = TilesetSpec {
        name: product_id.path(),
        base_z,
        grid_spec: grid,
        color_map: None,
        reducer: Reducer::from(product_id.aggregation()),
        fill: vec![None; product_data.band_specs.len()],
        quantize: vec![None; product_data.band_specs.len()],
        omit_zero: vec![None; product_data.band_specs.len()],
        omit_class: vec![None; product_data.band_specs.len()],
        band_specs: product_data.band_specs,
        bounds: [min_lng, min_lat, max_lng, max_lat],
    };

    Ok(PreparedProduct {
        product_id,
        spec,
        chunks,
        pyramid: None,
    })
}

#[cfg(test)]
mod tests {
    use gpv_products::{
        model::{BandSpec, LngLatGrid},
        products::{
            FixedSurface, GeneratingProcessType, GpvProductElement, GpvProductIdentifier,
            PointValue, ProductData,
        },
    };
    use jiff::Timestamp;

    use super::*;

    #[test]
    fn prepared_tiles_keep_original_grid_coordinates() {
        let datetime = Timestamp::from_second(1_570_871_100).unwrap();
        let product_id = GpvProductIdentifier {
            kind: GpvProductElement::HiresNowcastIntensity,
            datetime,
            reference_datetime: datetime,
            generating_process: GeneratingProcessType::Forecast,
            ..Default::default()
        };
        let product_data = ProductData {
            points: vec![PointValue {
                x: 100,
                y: 200,
                point_id: 0,
                point_power: 0,
                band_idx: 0,
                value: 42,
            }],
            band_specs: vec![BandSpec {
                name: "value".to_string(),
                ..Default::default()
            }],
            grid: None,
        };

        let prepared = prepare_product(product_id, product_data, false).unwrap();
        let mut position_count = 0;
        for (_, tile) in &prepared.chunks {
            assert_eq!(tile.point_positions.len(), tile.point_powers.len());
            for position in &tile.point_positions {
                assert_eq!(*position, (100, 200));
                position_count += 1;
            }
        }

        assert!(position_count > 0);
    }

    #[test]
    fn preparation_uses_the_grid_derived_from_the_message() {
        let grid = LngLatGrid {
            lng_0: 120.0,
            lat_0: 20.0,
            lng_denom: 40.0,
            lat_denom: 50.0,
        };
        let product_id = GpvProductIdentifier {
            kind: GpvProductElement::LfmWind,
            surface: FixedSurface::IsobaricSurface(1000, 0),
            generating_process: GeneratingProcessType::Forecast,
            ..Default::default()
        };
        let product_data = ProductData {
            points: vec![PointValue {
                x: 0,
                y: 0,
                point_id: 0,
                point_power: 0,
                band_idx: 0,
                value: 42,
            }],
            band_specs: vec![BandSpec::default(), BandSpec::default()],
            grid: Some(grid.clone()),
        };

        let prepared = prepare_product(product_id, product_data, false).unwrap();

        assert_eq!(prepared.spec.grid_spec, grid);
        assert_eq!(prepared.spec.base_z, 7);
    }
}
