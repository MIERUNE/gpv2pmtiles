//! Resolution of the Mercator lattice at each zoom.
//!
//! Detail is log2 of pixels across a tile: 8 is 256, 9 is 512.
//! The transition to source cells retains the original lattice zoom schedule.

use std::str::FromStr;

use anyhow::{Context, Error, Result, bail, ensure};

/// Log2 of the default 256 pixels across an MVT tile.
pub(crate) const DEFAULT_DETAIL: u8 = 8;

/// The coarsest lattice: 128 pixels across, the fewest at which the default
/// buffer of 2/256 of a tile still keeps one whole pixel of margin.
const MIN_DETAIL: u8 = 7;

/// Limit lattice tiles to 4096 pixels across.
const MAX_DETAIL: u8 = 12;

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
                (MIN_DETAIL..=MAX_DETAIL).contains(&detail),
                "--detail {detail} is outside {MIN_DETAIL} to {MAX_DETAIL}"
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

/// Whether MVT uses the lattice at this zoom. Preserve the original
/// transition, including the limit imposed by the source-cell extent.
pub(crate) fn uses_lattice(
    maximum_detail_zoom: u8,
    zoom: u8,
    detail: u8,
    grid: (f32, f32),
    extent: u32,
) -> bool {
    let requested = i32::from(maximum_detail_zoom) + i32::from(DEFAULT_DETAIL)
        - i32::from(detail)
        - i32::from(zoom);
    let cells_per_turn = f64::from(grid.0.abs().max(grid.1.abs())) * 360.0;
    let finest = (cells_per_turn / f64::from(extent) * (1.0 + 1e-9))
        .log2()
        .ceil() as i32
        - i32::from(zoom);
    requested > 0 || (finest > 0 && zoom < maximum_detail_zoom)
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
            "", "a", "1:", ":8", "5:8,1:9", "5:8,5:9", "1:9,8", "1:13", "1:31", "32:8", "6",
            "1:9,5:6",
        ] {
            assert!(
                spec.parse::<DetailSchedule>().is_err(),
                "{spec:?} was accepted"
            );
        }
    }

    #[test]
    fn lattice_transition_keeps_the_original_zoom_schedule() {
        for zoom in 0..=16 {
            assert_eq!(uses_lattice(9, zoom, 7, (320.0, 480.0), 4096), zoom < 10);
            assert_eq!(uses_lattice(9, zoom, 8, (320.0, 480.0), 4096), zoom < 9);
            assert_eq!(uses_lattice(9, zoom, 9, (320.0, 480.0), 4096), zoom < 8);
            assert_eq!(uses_lattice(9, zoom, 12, (320.0, 480.0), 4096), zoom < 6);
            assert_eq!(uses_lattice(9, zoom, 12, (320.0, 480.0), 256), zoom < 9);
        }
    }
}
