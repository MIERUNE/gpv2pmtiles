//! End-to-end checks exercise memberships, statistics, quantization and encoding.
use crate::tile::Reducer as Aggregation;
use crate::{
    model::PreparedProduct,
    prepare::{self, RawProduct},
    tile::{self, ColorEncoding, MvtLayout, RasterLayout, TileContext, TileOutput},
};
use gpv_products::{
    model::{BandSpec, LngLatGrid},
    products::{GpvProductElement, GpvProductIdentifier, PointValue, ProductData},
};

fn grid() -> LngLatGrid {
    LngLatGrid {
        lng_0: 0.1,
        lat_0: 0.1,
        lng_denom: 100.0,
        lat_denom: 100.0,
    }
}

fn raw(cells: &[(u16, u16, u8, i32)], grid: LngLatGrid) -> RawProduct {
    RawProduct {
        product_id: GpvProductIdentifier {
            kind: GpvProductElement::GsmTemperature,
            generating_process: gpv_products::products::GeneratingProcessType::Forecast,
            ..Default::default()
        },
        data: ProductData {
            points: cells
                .iter()
                .map(|&(x, y, point_power, value)| PointValue {
                    x,
                    y,
                    point_power,
                    value,
                    band_idx: 0,
                    point_id: 0,
                })
                .collect(),
            band_specs: vec![BandSpec {
                name: "value".into(),
                ..Default::default()
            }],
            grid: Some(grid),
        },
    }
}

fn product(
    cells: &[(u16, u16, u8, i32)],
    grid: LngLatGrid,
    method: Aggregation,
) -> PreparedProduct {
    let mut product = prepare::prepare_products(vec![raw(cells, grid)], true)
        .unwrap()
        .pop()
        .unwrap();
    product.spec.reducer = method;
    product
}

fn raster(product: &PreparedProduct, zxy: tile::Zxy, encoding: ColorEncoding) -> Vec<u8> {
    raster_with_size(product, zxy, RasterLayout::new(encoding))
}

fn raster_with_size(product: &PreparedProduct, zxy: tile::Zxy, layout: RasterLayout) -> Vec<u8> {
    let bytes = tile::make_layer(
        &TileContext::new(zxy),
        product,
        "test",
        TileOutput::Raster(layout),
    )
    .unwrap()
    .unwrap();
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let mut reader = decoder.read_info().unwrap();
    let mut bytes = vec![0; reader.output_buffer_size().unwrap()];
    reader.next_frame(&mut bytes).unwrap();
    bytes
}

fn mvt_value(product: &PreparedProduct) -> i64 {
    let bytes = tile::make_layer(
        &TileContext::new((0, 0, 0)),
        product,
        "test",
        TileOutput::Mvt(MvtLayout::default()),
    )
    .unwrap()
    .unwrap();
    let reader = fast_mvt::MvtReaderRef::new(&bytes).unwrap();
    let layer = reader.layers().next().unwrap();
    let features: Vec<_> = layer.features().collect();
    assert_eq!(features.len(), 1);
    match features[0].properties_vec().unwrap()[0].1 {
        fast_mvt::MvtValueRef::SInt(value) => value,
        other => panic!("{other:?}"),
    }
}

#[test]
fn vector_average_weights_mixed_sizes_and_deduplicates_chunk_copies() {
    let product = product(
        &[(0, 0, 1, 100), (2, 0, 0, 0)],
        LngLatGrid {
            lng_0: 0.0,
            ..grid()
        },
        Aggregation::Mean,
    );
    assert_eq!(mvt_value(&product), 80);
}

#[test]
fn vector_rounds_the_final_average_symmetrically() {
    for (a, b, expected) in [
        (0, 1, 0),
        (-1, 0, 0),
        (i32::MAX, i32::MAX, i64::from(i32::MAX)),
    ] {
        let product = product(&[(0, 0, 0, a), (1, 0, 0, b)], grid(), Aggregation::Mean);
        assert_eq!(mvt_value(&product), expected);
    }
}

#[test]
fn raster_average_keeps_fractional_raw_values_until_encoding() {
    let product = product(&[(0, 0, 0, 1), (1, 0, 0, 2)], grid(), Aggregation::Mean);
    let pixels = raster(
        &product,
        (0, 0, 0),
        ColorEncoding::Scalar24 {
            base: 0.0,
            interval: 0.01,
        },
    );
    let values: Vec<_> = pixels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| (u32::from(p[0]) << 16) | (u32::from(p[1]) << 8) | u32::from(p[2]))
        .filter(|&v| v != 0)
        .collect();
    assert!(!values.is_empty());
    assert!(values.iter().all(|&v| v == 150));
}

#[test]
fn raster_quantizes_and_omits_only_after_averaging() {
    let mut product = product(&[(0, 0, 0, 1), (1, 0, 0, 2)], grid(), Aggregation::Mean);
    product.spec.quantize =
        crate::quantize::resolve(&["0:7,1.75:9".into()], &product.spec.band_specs).unwrap();
    assert_eq!(mvt_value(&product), 7);
    let pixels = raster(
        &product,
        (0, 0, 0),
        ColorEncoding::Scalar24 {
            base: 0.0,
            interval: 1.0,
        },
    );
    assert!(
        pixels
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|p| p[2] != 0)
            .all(|p| p[2] == 7)
    );
    product.spec.omit_class = crate::quantize::resolve_class_omits(
        &["7".into()],
        &product.spec.quantize,
        &product.spec.band_specs,
    )
    .unwrap();
    assert!(
        tile::make_layer(
            &TileContext::new((0, 0, 0)),
            &product,
            "test",
            TileOutput::Raster(RasterLayout::new(ColorEncoding::Scalar24 {
                base: 0.0,
                interval: 1.0
            }))
        )
        .unwrap()
        .is_none()
    );
}

fn assert_horizontal(left: &[u8], right: &[u8]) {
    let size = 514;
    for row in 0..size {
        assert_eq!(
            &left[(row * size + size - 2) * 4..(row * size + size) * 4],
            &right[row * size * 4..(row * size + 2) * 4],
            "row={row}"
        );
    }
}

#[test]
fn raster_statistics_match_in_horizontal_and_vertical_tile_margins() {
    let cells: Vec<_> = (0..65)
        .flat_map(|x| (0..65).map(move |y| (x, y, 0, 10 + i32::from(x) + 2 * i32::from(y))))
        .collect();
    let grid = LngLatGrid {
        lng_0: -0.32,
        lat_0: -0.32,
        ..grid()
    };
    let size = 514;
    for method in [
        Aggregation::Legacy(gpv_products::model::Aggregation::Max),
        Aggregation::Mean,
        "p95".parse::<Aggregation>().unwrap(),
    ] {
        for fill in [None, Some(0)] {
            let mut product = product(&cells, grid.clone(), method);
            product.spec.fill[0] = fill;
            let encoding = ColorEncoding::Scalar16 {
                min: 0.0,
                max: 300.0,
            };
            let nw = raster(&product, (1, 0, 0), encoding);
            let ne = raster(&product, (1, 1, 0), encoding);
            let se = raster(&product, (1, 1, 1), encoding);
            assert_horizontal(&nw, &ne);
            assert_eq!(
                &ne[(size - 2) * size * 4..],
                &se[..2 * size * 4],
                "method={method:?}, fill={fill:?}"
            );
        }
    }
}

#[test]
fn smaller_tiles_have_identical_pixels_at_the_equivalent_zoom() {
    let cells: Vec<_> = (0..65)
        .flat_map(|x| (0..65).map(move |y| (x, y, 0, 10 + i32::from(x) + i32::from(y))))
        .collect();
    for denom in [100.0, 1.0] {
        for method in [Aggregation::Mean, "p95".parse().unwrap()] {
            let product = product(
                &cells,
                LngLatGrid {
                    lng_0: -0.32,
                    lat_0: -0.32,
                    lng_denom: denom,
                    lat_denom: denom,
                },
                method,
            );
            let encoding = ColorEncoding::Scalar16 {
                min: 0.0,
                max: 300.0,
            };
            let large = raster(&product, (1, 1, 0), encoding);
            let small = raster_with_size(
                &product,
                (2, 2, 1),
                RasterLayout {
                    encoding,
                    size: 256,
                },
            );
            for row in 0..258 {
                assert_eq!(
                    &large[((row + 256) * 514) * 4..((row + 256) * 514 + 258) * 4],
                    &small[(row * 258) * 4..((row + 1) * 258) * 4],
                    "row={row}, denom={denom}, method={method:?}"
                );
            }
        }
    }
}

#[test]
fn coarse_raster_cells_match_expansion_into_basic_grid_cells() {
    let mixed = [(0, 0, 2, 100), (4, 0, 0, 30), (6, 4, 0, -5)];
    let mut expanded: Vec<_> = (0..4)
        .flat_map(|x| (0..4).map(move |y| (x, y, 0, 100)))
        .collect();
    expanded.extend_from_slice(&mixed[1..]);
    let grid = LngLatGrid {
        lng_0: -0.015,
        lat_0: -0.015,
        ..grid()
    };
    let encoding = ColorEncoding::Scalar16 {
        min: -10.0,
        max: 110.0,
    };
    for method in [
        Aggregation::Mean,
        "max".parse().unwrap(),
        "p95".parse().unwrap(),
    ] {
        for fill in [None, Some(0)] {
            let mut mixed = product(&mixed, grid.clone(), method);
            let mut expanded = product(&expanded, grid.clone(), method);
            mixed.spec.fill[0] = fill;
            expanded.spec.fill[0] = fill;
            assert_eq!(
                raster(&mixed, (6, 32, 31), encoding),
                raster(&expanded, (6, 32, 31), encoding),
                "method={method:?}, fill={fill:?}"
            );
        }
    }
}

#[test]
fn raster_statistics_wrap_at_the_antimeridian() {
    let cells: Vec<_> = (0..65)
        .flat_map(|x| (0..65).map(move |y| (x, y, 0, 10 + i32::from(x) + i32::from(y))))
        .collect();
    let grid = LngLatGrid {
        lng_0: 179.68,
        lat_0: -0.32,
        ..grid()
    };
    for method in [
        Aggregation::Legacy(gpv_products::model::Aggregation::Max),
        Aggregation::Mean,
        "p95".parse::<Aggregation>().unwrap(),
    ] {
        let product = product(&cells, grid.clone(), method);
        let encoding = ColorEncoding::Scalar16 {
            min: 0.0,
            max: 300.0,
        };
        assert_horizontal(
            &raster(&product, (1, 1, 0), encoding),
            &raster(&product, (1, 0, 0), encoding),
        );
    }
}

#[test]
fn preparation_rejects_overlap_misalignment_and_conflicting_values() {
    for cells in [
        vec![(0, 0, 2, 10), (1, 1, 0, 20)],
        vec![(1, 0, 1, 10)],
        vec![(0, 0, 0, 10), (0, 0, 0, 20)],
        vec![(0, 0, 17, 10)],
    ] {
        assert!(
            prepare::prepare_products(vec![raw(&cells, grid())], true).is_err(),
            "{cells:?}"
        );
    }
    assert!(
        prepare::prepare_products(vec![raw(&[(0, 0, 1, 10), (2, 0, 0, 20)], grid())], true).is_ok()
    );
    // A maximum gives the same answer however the cells lie, so input that
    // converted before is not refused unless a weighted statistic needs it.
    for cells in [
        vec![(0, 0, 2, 10), (1, 1, 0, 20)],
        vec![(1, 0, 1, 10)],
        vec![(0, 0, 0, 10), (0, 0, 0, 20)],
    ] {
        assert!(
            prepare::prepare_products(vec![raw(&cells, grid())], false).is_ok(),
            "{cells:?}"
        );
    }
}

#[test]
fn omitted_percentile_blocks_do_not_leave_attribute_less_vector_polygons() {
    let mut product = product(
        &[(0, 0, 0, 10)],
        grid(),
        "p50".parse::<Aggregation>().unwrap(),
    );
    product.spec.fill[0] = Some(0);
    product.spec.omit_zero = crate::quantize::resolve_zero_omits(true, &product.spec.band_specs);
    assert!(
        tile::make_layer(
            &TileContext::new((0, 0, 0)),
            &product,
            "test",
            TileOutput::Mvt(MvtLayout::default())
        )
        .unwrap()
        .is_none()
    );
}

#[test]
fn source_overlap_validation_matches_rectangle_intersections() {
    let mut seed = 0x1357_2468_abcd_ef01_u64;
    let mut random = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for _ in 0..2000 {
        let cells: Vec<_> = (0..(1 + random() % 20))
            .map(|_| {
                let power = (random() % 4) as u8;
                let mask = !((1u16 << power) - 1);
                (
                    (random() as u16 % 64) & mask,
                    (random() as u16 % 64) & mask,
                    power,
                    10,
                )
            })
            .collect();
        let mut overlapping = false;
        for (i, &(x, y, p, _)) in cells.iter().enumerate() {
            for &(a, b, q, _) in &cells[..i] {
                if (x, y, p) == (a, b, q) {
                    continue;
                }
                overlapping |= u32::from(x) < u32::from(a) + (1 << q)
                    && u32::from(a) < u32::from(x) + (1 << p)
                    && u32::from(y) < u32::from(b) + (1 << q)
                    && u32::from(b) < u32::from(y) + (1 << p);
            }
        }
        assert_eq!(
            prepare::prepare_products(vec![raw(&cells, grid())], true).is_err(),
            overlapping,
            "{cells:?}"
        );
    }
}

#[test]
fn remote_support_does_not_change_local_statistics() {
    let local: Vec<_> = (0..65)
        .flat_map(|x| {
            (6000..6065).map(move |y| (x, y, 0, 1 + (i32::from(x) * 17 + i32::from(y) * 31) % 100))
        })
        .collect();
    let mut remote = local.clone();
    remote.push((0, 0, 0, 1));
    let grid = LngLatGrid {
        lng_0: 0.1,
        lat_0: 0.1,
        lng_denom: 80.,
        lat_denom: 120.,
    };
    let near = product(&local, grid.clone(), Aggregation::Mean);
    let far = product(&remote, grid, Aggregation::Mean);
    let encoding = ColorEncoding::Scalar16 { min: 0., max: 100. };
    let lat = (near.spec.bounds[1] + near.spec.bounds[3]) / 2.;
    let y = (crate::geo::lat_to_web_mercator_y(lat) * 64.).floor() as u32;
    let a = raster(&near, (6, 32, y), encoding);
    let b = raster(&far, (6, 32, y), encoding);
    let differences = a
        .as_chunks::<4>()
        .0
        .iter()
        .zip(b.as_chunks::<4>().0)
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "remote_support: near_bounds={:?}, far_bounds={:?}, changed_pixels={differences}",
        near.spec.bounds, far.spec.bounds
    );
    assert_eq!(
        differences, 0,
        "adding a remote cell changed local raster statistics"
    );
}

#[test]
fn fractional_physical_zero_is_omitted() {
    let mut product = product(&[(0, 0, 0, -1), (1, 0, 0, 0)], grid(), Aggregation::Mean);
    product.spec.band_specs[0].reference_value = 0.5;
    product.spec.omit_zero = crate::quantize::resolve_zero_omits(true, &product.spec.band_specs);
    let tile = tile::make_layer(
        &TileContext::new((0, 0, 0)),
        &product,
        "test",
        TileOutput::Raster(RasterLayout::new(ColorEncoding::Scalar16 {
            min: -1.,
            max: 1.,
        })),
    )
    .unwrap();
    eprintln!(
        "fractional_zero: zero_omit={:?}, raster_present={}",
        product.spec.omit_zero,
        tile.is_some()
    );
    assert!(
        tile.is_none(),
        "the physical mean is zero, but --omit-zero left it in the raster"
    );
}

#[test]
fn missing_band_is_excluded_instead_of_zero_filled() {
    let mut input = raw(&[(0, 0, 0, 10), (1, 0, 0, 20)], grid());
    input.product_id.kind = GpvProductElement::GsmWind;
    input.data.band_specs = vec![
        BandSpec {
            name: "u".into(),
            ..Default::default()
        },
        BandSpec {
            name: "v".into(),
            ..Default::default()
        },
    ];
    input.data.points.push(PointValue {
        x: 1,
        y: 0,
        point_power: 0,
        value: 30,
        band_idx: 1,
        point_id: 0,
    });
    let mut product = prepare::prepare_products(vec![input], true)
        .unwrap()
        .pop()
        .unwrap();
    product.spec.reducer = Aggregation::Mean;
    let bytes = tile::make_layer(
        &TileContext::new((0, 0, 0)),
        &product,
        "test",
        TileOutput::Mvt(MvtLayout::default()),
    )
    .unwrap()
    .unwrap();
    let reader = fast_mvt::MvtReaderRef::new(&bytes).unwrap();
    let layer = reader.layers().next().unwrap();
    let features: Vec<_> = layer.features().collect();
    eprintln!(
        "partial_vector: MVT={:?}",
        features[0].properties_vec().unwrap()
    );
    let pixels = raster(
        &product,
        (0, 0, 0),
        ColorEncoding::VectorField {
            component_limit: 100.,
        },
    );
    let pixel = pixels
        .as_chunks::<4>()
        .0
        .iter()
        .find(|p| p[2] == 255)
        .unwrap();
    let raster_v = (f64::from(pixel[1]) - 128.) * 100. / 127.;
    eprintln!("partial_vector: raster_v={raster_v}, expected_missing_excluded_v=30");
    for (_, chunk) in &product.chunks {
        for (i, xy) in chunk.point_positions.iter().enumerate() {
            if *xy == (0, 0) {
                eprintln!(
                    "partial_vector: missing v after prepare={:?}",
                    chunk.bands[1].values[i].get()
                );
            }
        }
    }
    assert!(
        (raster_v - 30.).abs() < 0.8,
        "a missing band became zero before aggregation"
    );
}

fn wrapped_product(
    denom: f32,
    columns: std::ops::Range<u16>,
    method: Aggregation,
) -> PreparedProduct {
    let cells: Vec<_> = columns
        .flat_map(|x| (0..200).map(move |y| (x, y, 0, 100)))
        .collect();
    let mut p = product(
        &cells,
        LngLatGrid {
            lng_0: 0.,
            lat_0: -1.,
            lng_denom: denom,
            lat_denom: 100.,
        },
        method,
    );
    p.spec.fill[0] = Some(0);
    p
}

#[test]
fn wrapped_constant_field_has_constant_mean() {
    let product = wrapped_product(3.3333333, 890..910, Aggregation::Mean);
    let frame = raster(
        &product,
        (0, 0, 0),
        ColorEncoding::Scalar24 {
            base: 0.,
            interval: 0.01,
        },
    );
    let offset = (257 * 514 + 128) * 4;
    let px = &frame[offset..offset + 4];
    let code = u32::from(px[0]) * 65536 + u32::from(px[1]) * 256 + u32::from(px[2]);
    eprintln!("wrapped_constant: decoded_mean={}", f64::from(code) * 0.01);
    assert_eq!(
        code, 10000,
        "a constant field changed under longitude wrapping"
    );
}

#[test]
fn wrapped_constant_field_percentile_does_not_panic() {
    let product = wrapped_product(1_000_000.0 / 9990.0, 26950..27110, "p95".parse().unwrap());
    let _frame = raster(
        &product,
        (0, 0, 0),
        ColorEncoding::Scalar24 {
            base: 0.,
            interval: 0.01,
        },
    );
}
