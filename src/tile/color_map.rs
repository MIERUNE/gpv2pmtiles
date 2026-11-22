//! Fixed RGBA colours for intervals of physical or quantized output values.

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ColorMap {
    stops: Vec<(f64, [u8; 4])>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ColorStop {
    lower: f64,
    color: String,
    opacity: Option<f64>,
}

fn parse_color(color: &str) -> Result<[u8; 4]> {
    let hex = color
        .trim()
        .strip_prefix('#')
        .context("colour-map colours must begin with #")?;
    ensure!(
        matches!(hex.len(), 6 | 8) && hex.bytes().all(|b| b.is_ascii_hexdigit()),
        "colour-map colours must be #RRGGBB or #RRGGBBAA"
    );
    let mut rgba = [0, 0, 0, 255];
    for (index, component) in rgba.iter_mut().enumerate().take(hex.len() / 2) {
        *component = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16)?;
    }
    Ok(rgba)
}

impl std::str::FromStr for ColorMap {
    type Err = anyhow::Error;

    fn from_str(spec: &str) -> Result<Self> {
        if let Some(path) = spec.strip_prefix('@') {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("cannot read colour map {path:?}"))?;
            return Self::from_json(&content)
                .with_context(|| format!("invalid colour map {path:?}"));
        }
        let mut stops = Vec::new();
        for entry in spec.split(',') {
            let (lower, color) = entry.trim().split_once(':').context(
                "--color-map expects <lower>:#RRGGBB or <lower>:#RRGGBBAA, separated by commas, or @FILE.json",
            )?;
            let lower = lower
                .trim()
                .parse::<f64>()
                .context("invalid colour-map lower bound")?;
            stops.push((lower, parse_color(color)?));
        }
        Self::new(stops)
    }
}

impl ColorMap {
    fn new(stops: Vec<(f64, [u8; 4])>) -> Result<Self> {
        ensure!(
            !stops.is_empty(),
            "colour map must contain at least one stop"
        );
        ensure!(
            stops.iter().all(|stop| stop.0.is_finite()),
            "colour-map bounds must be finite"
        );
        ensure!(
            stops.windows(2).all(|pair| pair[1].0 > pair[0].0),
            "colour-map bounds must strictly increase"
        );
        Ok(Self { stops })
    }

    fn from_json(content: &str) -> Result<Self> {
        let entries: Vec<ColorStop> = serde_json::from_str(content)
            .context("expected an array of {lower, color, opacity?}")?;
        let mut stops = Vec::with_capacity(entries.len());
        for entry in entries {
            let mut rgba = parse_color(&entry.color)?;
            if let Some(opacity) = entry.opacity {
                ensure!(
                    opacity.is_finite() && (0.0..=1.0).contains(&opacity),
                    "colour-map opacity must be between 0 and 1"
                );
                ensure!(
                    entry.color.trim().len() == 7,
                    "use either #RRGGBBAA or opacity for a colour, not both"
                );
                rgba[3] = (opacity * 255.0).round() as u8;
            }
            stops.push((entry.lower, rgba));
        }
        Self::new(stops)
    }

    pub(crate) fn paint(&self, value: f64) -> [u8; 4] {
        if !value.is_finite() {
            return [0; 4];
        }
        let end = self.stops.partition_point(|(lower, _)| *lower <= value);
        end.checked_sub(1)
            .map_or([0; 4], |index| self.stops[index].1)
    }

    pub(crate) fn descriptor(&self) -> Value {
        json!({
            "kind": "classified", "bounds": "inclusive-lower", "below_first": "transparent",
            "missing": "transparent", "alpha": "straight",
            "input": "physical-or-quantized-output-value",
            "stops": self.stops.iter().map(|(lower, rgba)| json!({"lower": lower, "rgba": rgba, "opacity": f64::from(rgba[3]) / 255.0})).collect::<Vec<_>>()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_edges_alpha_and_missing_values() {
        let map: ColorMap = "-1:#112233,0.2:#44556680,80:#aAbBcC".parse().unwrap();
        for value in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY, -1.001] {
            assert_eq!(map.paint(value), [0; 4]);
        }
        assert_eq!(map.paint(-1.0), [17, 34, 51, 255]);
        assert_eq!(map.paint(0.19999), [17, 34, 51, 255]);
        assert_eq!(map.paint(0.2), [68, 85, 102, 128]);
        assert_eq!(map.paint(80.0), [170, 187, 204, 255]);
        assert_eq!(map.paint(1000.0), [170, 187, 204, 255]);
    }

    #[test]
    fn json_stops_have_individual_opacity() {
        let map = ColorMap::from_json(
            r##"[
            {"lower":0.2,"color":"#C8E6FF","opacity":0},
            {"lower":1,"color":"#64B4FF","opacity":0.5},
            {"lower":2,"color":"#1E64E6","opacity":1},
            {"lower":4,"color":"#0A3CB440"}
        ]"##,
        )
        .unwrap();
        assert_eq!(map.paint(0.2), [200, 230, 255, 0]);
        assert_eq!(map.paint(1.0), [100, 180, 255, 128]);
        assert_eq!(map.paint(2.0), [30, 100, 230, 255]);
        assert_eq!(map.paint(4.0), [10, 60, 180, 64]);
        for spec in [
            r##"[{"lower":0,"color":"#ffffff","opacity":-0.1}]"##,
            r##"[{"lower":0,"color":"#ffffff","opacity":1.1}]"##,
            r##"[{"lower":0,"color":"#ffffff80","opacity":0.5}]"##,
            r##"[{"lower":0,"color":"#ffffff","opacty":0.5}]"##,
            "[]",
        ] {
            assert!(ColorMap::from_json(spec).is_err(), "accepted {spec}");
        }
    }

    #[test]
    fn a_file_reads_as_its_json() {
        let content =
            r##"[{"lower":0,"color":"#102030"},{"lower":5,"color":"#405060","opacity":0.5}]"##;
        let path = std::env::temp_dir().join(format!("color-map-{}.json", std::process::id()));
        std::fs::write(&path, content).unwrap();
        let from_file: ColorMap = format!("@{}", path.display()).parse().unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(from_file, ColorMap::from_json(content).unwrap());
        assert_eq!(from_file.paint(5.0), [64, 80, 96, 128]);
    }

    #[test]
    fn invalid_maps_are_rejected() {
        for spec in [
            "",
            "0:#000000,",
            "NaN:#000000",
            "inf:#000000",
            "0:#000000,0:#ffffff",
            "1:#000000,0:#ffffff",
            "0:red",
            "0:#fff",
            "0:#00000g",
            "0:#あいう",
            "0:#000000000",
        ] {
            assert!(spec.parse::<ColorMap>().is_err(), "accepted {spec}");
        }
    }
}
