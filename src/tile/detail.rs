//! How coarsely grid cells are merged at each zoom.
//!
//! A cell much smaller than what a tile can show only adds geometry, so the
//! cells of a low zoom are merged into blocks before the polygons are built,
//! with the block's value aggregated from the cells inside it. The detail of a
//! zoom is the log2 of the resolution the blocks should approach: at detail 8
//! a block is about 1/256 of the tile across, at 9 about 1/512. This is the
//! same scale as an MVT extent, so a tiling recipe that lowers the extent at
//! low zooms translates into a detail per zoom, but here it coarsens the grid
//! rather than the tile coordinates, and the extent stays whatever it is. The
//! extent does bound the detail from above: a block narrower than one of its
//! units would be rounded away, so raising the detail stops short of that
//! (see [`mvt_aggregation_scale`]). The default blocks are kept as they are
//! even where a small extent already makes them narrower than a unit.
//!
//! Blocks are a power of two cells across, so the size reached is the nearest
//! one to the request rather than the request itself.
//!
//! The detail can only be raised above the default, never lowered. A block
//! larger than the default reaches past the margin of neighbouring cells a
//! tile reads, so the copies of a block on either side of a tile edge would be
//! aggregated from different cells and disagree at the seam; and averaging
//! products sum a block's raw values in an i32, which a block of a whole
//! grid can overflow.

use std::str::FromStr;

use anyhow::{Context, Error, Result, bail, ensure};

/// The detail every zoom used before it could be chosen, which keeps the
/// output of a run that does not choose unchanged.
pub(crate) const DEFAULT_DETAIL: u8 = 8;

/// Beyond this a block would be smaller than a cell at every zoom a grid is
/// tiled to.
const MAX_DETAIL: u8 = 30;

/// Zooms past this are not addressable with the u32 tile coordinates used
/// elsewhere.
const MAX_ZOOM: u8 = 31;

/// Detail for every zoom, from stops of the form `zoom:detail`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DetailSchedule([u8; MAX_ZOOM as usize + 1]);

impl Default for DetailSchedule {
    fn default() -> Self {
        Self([DEFAULT_DETAIL; MAX_ZOOM as usize + 1])
    }
}

impl DetailSchedule {
    pub(crate) fn at(&self, zoom: u8) -> u8 {
        self.0[usize::from(zoom.min(MAX_ZOOM))]
    }
}

impl FromStr for DetailSchedule {
    type Err = Error;

    /// Parses `8` for one detail everywhere, or `1:9,5:8,9:12` for detail 9
    /// from zoom 1, 8 from zoom 5 and 12 from zoom 9. As with the boundaries of
    /// `--quantize`, zooms before the first stop take its detail.
    fn from_str(spec: &str) -> Result<Self> {
        let invalid =
            || format!("--detail expects <detail> or <zoom>:<detail>,... but found {spec:?}");
        let mut stops = Vec::new();
        for stop in spec.split(',') {
            let stop = stop.trim();
            let (zoom, detail) = match stop.split_once(':') {
                Some((zoom, detail)) => (
                    zoom.trim().parse::<u8>().with_context(invalid)?,
                    detail.trim().parse::<u8>().with_context(invalid)?,
                ),
                None if spec.split(',').count() == 1 => {
                    (0, stop.parse::<u8>().with_context(invalid)?)
                }
                None => bail!("{}", invalid()),
            };
            ensure!(
                zoom <= MAX_ZOOM,
                "--detail zoom {zoom} is beyond {MAX_ZOOM}"
            );
            ensure!(
                (DEFAULT_DETAIL..=MAX_DETAIL).contains(&detail),
                "--detail {detail} is outside {DEFAULT_DETAIL} to {MAX_DETAIL}; it can only make blocks finer than the default"
            );
            if let Some(&(previous, _)) = stops.last() {
                ensure!(
                    zoom > previous,
                    "--detail zooms must increase, but {zoom} follows {previous}"
                );
            }
            stops.push((zoom, detail));
        }

        let mut schedule = [stops[0].1; MAX_ZOOM as usize + 1];
        for (index, &(zoom, detail)) in stops.iter().enumerate() {
            let end = stops
                .get(index + 1)
                .map_or(usize::from(MAX_ZOOM) + 1, |&(next, _)| usize::from(next));
            schedule[usize::from(zoom)..end].fill(detail);
        }
        Ok(Self(schedule))
    }
}

/// How many times the cells are halved into blocks at `zoom`, as a power of
/// two: a block is `1 << scale` cells across.
///
/// `maximum_detail_zoom` is the zoom at which one cell is about 1/256 of a
/// tile, so at detail 8 the blocks keep that size by doubling for every zoom
/// out. Each step of detail moves the size by the same factor of two that a
/// step of zoom does.
pub(crate) fn aggregation_scale(maximum_detail_zoom: u8, zoom: u8, detail: u8) -> u8 {
    let scale = i32::from(maximum_detail_zoom) + i32::from(DEFAULT_DETAIL)
        - i32::from(detail)
        - i32::from(zoom);
    scale.max(0) as u8
}

/// The smallest scale at which a block is at least one unit of `extent` across
/// in both directions.
///
/// The polygons are snapped to whole units of the tile, and the grid of
/// cells does not line up with them, so a block narrower than a unit can have
/// both of its edges rounded to the same unit and vanish from the tile,
/// value and all. One unit or more always keeps two distinct edges. Mercator
/// stretches latitude away from the equator, so a block is never shorter
/// than it would be there, which is the height assumed.
pub(crate) fn finest_scale(lng_denom: f32, lat_denom: f32, zoom: u8, extent: u32) -> u8 {
    let cells_per_turn = f64::from(lng_denom.abs().max(lat_denom.abs())) * 360.0;
    // A sliver above the exact quotient keeps a block that comes out at
    // precisely one unit from being rounded below it.
    let needed = (cells_per_turn / f64::from(extent) * (1.0 + 1e-9))
        .log2()
        .ceil() as i32;
    (needed - i32::from(zoom)).max(0) as u8
}

/// The scale the cells of `zoom` are merged at in an MVT tile of `extent`.
///
/// The detail asks for blocks finer than the default; this keeps them from
/// becoming so fine that they fall below one unit. It never merges more
/// coarsely than the default does, since larger blocks reach past the cells
/// a tile reads around itself - an extent too small for the default blocks
/// keeps the default and its rounding.
pub(crate) fn mvt_aggregation_scale(
    maximum_detail_zoom: u8,
    zoom: u8,
    detail: u8,
    grid: (f32, f32),
    extent: u32,
) -> u8 {
    let requested = aggregation_scale(maximum_detail_zoom, zoom, detail);
    let default = aggregation_scale(maximum_detail_zoom, zoom, DEFAULT_DETAIL);
    requested.max(finest_scale(grid.0, grid.1, zoom, extent).min(default))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_detail_applies_to_every_zoom() {
        let schedule = "9".parse::<DetailSchedule>().unwrap();

        assert!((0..=MAX_ZOOM).all(|zoom| schedule.at(zoom) == 9));
    }

    #[test]
    fn stops_hold_until_the_next_one() {
        // The extent switch of a tiling recipe: 512 below zoom 5, 256 below 9,
        // and 4096 from there.
        let schedule = "1:9,5:8,9:12".parse::<DetailSchedule>().unwrap();

        let details = (0..=10).map(|zoom| schedule.at(zoom)).collect::<Vec<_>>();
        assert_eq!(details, [9, 9, 9, 9, 9, 8, 8, 8, 8, 12, 12]);
        assert_eq!(schedule.at(MAX_ZOOM), 12);
        assert_eq!(schedule.at(u8::MAX), 12);
    }

    #[test]
    fn malformed_schedules_are_refused() {
        for spec in [
            "", "a", "1:", ":8", "5:8,1:9", "5:8,5:9", "1:9,8", "1:31", "32:8", "7", "1:9,5:7",
        ] {
            assert!(
                spec.parse::<DetailSchedule>().is_err(),
                "{spec:?} was accepted"
            );
        }
    }

    #[test]
    fn the_default_detail_keeps_the_previous_aggregation() {
        for maximum_detail_zoom in 0..=16 {
            for zoom in 0..=24 {
                assert_eq!(
                    aggregation_scale(maximum_detail_zoom, zoom, DEFAULT_DETAIL),
                    maximum_detail_zoom.saturating_sub(zoom),
                    "maximum detail zoom {maximum_detail_zoom}, zoom {zoom}"
                );
            }
        }
    }

    #[test]
    fn blocks_never_fall_below_one_unit() {
        // The 250 m nowcast: 320 columns and 480 rows a degree. At extent 4096
        // a block has to be 2^6 cells across from zoom 0, so at zoom 5 the
        // detail 12 that would leave cells unmerged is held at two cells.
        assert_eq!(finest_scale(320.0, 480.0, 5, 4096), 1);
        assert_eq!(finest_scale(320.0, 480.0, 6, 4096), 0);
        assert_eq!(mvt_aggregation_scale(9, 5, 12, (320.0, 480.0), 4096), 1);
        // Where the detail asks for no less, it is left alone.
        assert_eq!(mvt_aggregation_scale(9, 4, 9, (320.0, 480.0), 4096), 4);
        // The rows are the finer direction here; the columns decide when
        // they are, and the sign of a southward grid does not matter.
        assert_eq!(finest_scale(480.0, -320.0, 5, 4096), 1);
    }

    #[test]
    fn the_default_is_kept_even_when_the_extent_is_too_small_for_it() {
        // At extent 256 one unit is about 2.6 default blocks of the nowcast,
        // but merging further would reach past the cells a tile reads.
        assert_eq!(
            mvt_aggregation_scale(9, 5, DEFAULT_DETAIL, (320.0, 480.0), 256),
            aggregation_scale(9, 5, DEFAULT_DETAIL)
        );
        assert_eq!(
            mvt_aggregation_scale(9, 5, 12, (320.0, 480.0), 256),
            aggregation_scale(9, 5, DEFAULT_DETAIL)
        );
    }

    #[test]
    fn the_default_detail_is_never_moved_at_the_default_extent() {
        // Every grid the products define, at every zoom it is tiled to.
        for (lng_denom, lat_denom) in [
            (320.0, 480.0),
            (80.0, 120.0),
            (16.0, 20.0),
            (8.0, 10.0),
            (2.0, 2.5),
            (1.0, 1.25),
            (0.8, 0.8),
        ] {
            let maximum_detail_zoom =
                (f64::from(lat_denom) * 360.0 * 2.0 / 512.0).log2().round() as u8;
            for zoom in 0..=16 {
                assert_eq!(
                    mvt_aggregation_scale(
                        maximum_detail_zoom,
                        zoom,
                        DEFAULT_DETAIL,
                        (lng_denom, lat_denom),
                        4096
                    ),
                    aggregation_scale(maximum_detail_zoom, zoom, DEFAULT_DETAIL),
                    "grid {lng_denom}x{lat_denom}, zoom {zoom}"
                );
            }
        }
    }

    #[test]
    fn each_step_of_detail_halves_the_blocks() {
        assert_eq!(aggregation_scale(9, 4, 8), 5);
        assert_eq!(aggregation_scale(9, 4, 9), 4);
        // Finer than the grid itself leaves the cells as they are.
        assert_eq!(aggregation_scale(9, 9, 12), 0);
        assert_eq!(aggregation_scale(9, 4, 30), 0);
    }
}
