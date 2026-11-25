use std::{
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use clap::{
    ArgAction, ArgMatches, CommandFactory, FromArgMatches, Parser, error::ErrorKind,
    parser::ValueSource,
};
use gpv2pmtiles::{ConvertOptions, convert, remove_unfinished_outputs};
use serde_json::Value;

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Cli {
    /// Input GRIB2 file. Gzip-compressed input is detected from the .gz suffix.
    input: PathBuf,

    /// Output PMTiles archive. With --raster and several forecast times, this
    /// must contain {seq} or {valid_time}, such as wind_{valid_time}.pmtiles.
    output: PathBuf,

    /// Take option defaults from a JSON file holding one object, which maps
    /// long option names without the dashes to values, such as
    /// {"product": "hrnowc/intensity", "max-zoom": 9, "omit-zero": true}.
    /// A flag takes true or false, and an option that can be repeated takes
    /// an array as well. A "description" key is ignored. Options given on the
    /// command line take precedence over the file.
    #[arg(long, value_name = "FILE")]
    profile: Option<PathBuf>,

    /// Product to convert, such as hrnowc/intensity. Required when the input contains several products.
    #[arg(long)]
    product: Option<String>,

    /// MVT source-layer name pattern. Must contain the {seq} placeholder when
    /// several layers are written; a single layer may use a fixed name.
    #[arg(long, default_value = "layer_{seq}")]
    layer_name_pattern: String,

    /// Keep only the first N value products after sorting by valid time.
    #[arg(long)]
    layer_count: Option<usize>,

    /// First number substituted for {seq}, so the numbering can match the
    /// forecast hours of the input. For example --layer-seq-start 1 turns an
    /// FH01-06 input into layers 1 through 6.
    #[arg(long, default_value_t = 0)]
    layer_seq_start: usize,

    /// Minimum output zoom.
    #[arg(long, default_value_t = 0)]
    min_zoom: u8,

    /// Maximum output zoom. By default it is derived from the source grid.
    #[arg(long)]
    max_zoom: Option<u8>,

    /// MVT tile extent, the number of units across a tile. [default: 4096]
    #[arg(long)]
    extent: Option<u32>,

    /// MVT geometry kept beyond each tile edge, in 1/256ths of the tile width
    /// (the pixels of a 256px tile, as in tippecanoe). With --extent 256 one
    /// pixel is one tile unit. [default: 2]
    #[arg(long, value_name = "PIXELS")]
    buffer: Option<u32>,

    /// MVT pixels across a low-zoom tile, as log2: 8 means 256, 9 means 512.
    /// Give one value for every zoom, or stops such as 1:9,5:8,9:12.
    /// Accepts 7 through 12. Low zooms use one MVT unit per pixel; higher
    /// zooms keep source cells at --extent. [default: 8]
    #[arg(long, value_name = "SPEC")]
    detail: Option<String>,

    /// Drop products that are analyses rather than forecasts. Nowcast inputs
    /// lead with the observed field, valid at the reference time itself, which
    /// otherwise becomes the first layer.
    #[arg(long)]
    skip_analysis: bool,

    /// Keep only products at least this many minutes ahead of the reference
    /// time. The observed field of a nowcast is valid at the reference time, so
    /// --min-lead-time 5 starts the layers at the first forecast step.
    #[arg(long, value_name = "MINUTES")]
    min_lead_time: Option<i64>,

    /// Rename a band as <FROM>=<TO>, such as --rename value=DN. The name is
    /// used as the MVT attribute key and the metadata field name, and
    /// --quantize refers to the new name. Repeat for several bands.
    #[arg(long, value_name = "FROM=TO")]
    rename: Vec<String>,

    /// Quantize values into classes before building polygons, which merges
    /// neighbouring cells and shrinks the geometry.
    ///
    /// Boundaries are inclusive lower bounds in physical units, optionally
    /// followed by the value the class should emit: --quantize "0,1,2,4,8" or
    /// --quantize "0:0,1:0.5,2:1.5". The last class is open ended, and values
    /// below the first boundary join the first class. Prefix with a band name
    /// and repeat the option when the product has several bands, such as
    /// --quantize "u=-50,0,50".
    #[arg(long, value_name = "SPEC")]
    quantize: Vec<String>,

    /// Leave cells whose quantized class emits one of these values out of the
    /// tile entirely, geometry included. Requires --quantize. Prefix with a
    /// band name when the product has several bands, and repeat as needed.
    #[arg(long, value_name = "VALUES")]
    omit_class: Vec<String>,

    /// Before quantization, leave cells whose physical value is exactly zero
    /// out of the tile on every band. A band that cannot represent zero has no
    /// cell to leave out, but a weighted mean of it can still come out at
    /// zero, and is left out then.
    #[arg(long)]
    omit_zero: bool,

    /// Override how cells are combined into a low-zoom pixel: max, min,
    /// mean, bit-or or p<percent>. Mean and percentiles weigh each cell by
    /// its area on the grid:
    ///
    ///   mean (avg)  the grid-area-weighted mean; vector raw values round half to even
    ///   p<percent>    the smallest value covering at least that share of
    ///                 the pixel, without interpolation, such as p95 or p99.5
    ///
    /// Weighted statistics fill unstored cells with zero for precipitation
    /// and snow, and exclude them otherwise. Override with --missing or --fill.
    #[arg(long, value_name = "METHOD", verbatim_doc_comment)]
    aggregation: Option<String>,

    /// Physical value counted over the part of a pixel no cell is stored for,
    /// for mean or p<percent>, including a product's default mean, overriding
    /// --missing for a band.
    /// Precipitation and snow already default to zero fill. The value must be one
    /// the band can hold exactly. Prefix with a band name when the product has
    /// several bands, and repeat as needed. A pixel with no stored cell stays
    /// empty.
    #[arg(long, value_name = "VALUE")]
    fill: Vec<String>,

    /// Weighted statistics: exclude missing cells or fill with a physical value.
    /// Defaults to zero for precipitation/snow, exclusion for other products.
    #[arg(long, value_name = "POLICY", allow_hyphen_values = true)]
    missing: Option<String>,

    /// Write RGBA PNG raster tiles instead of MVT. Choose classified display
    /// colours, or encode values into channels for decoding by a shader.
    ///
    /// The parameters belong to the archive rather than to a tile, so that
    /// every tile decodes the same way and the field does not jump at a seam.
    /// The channels, offset, scale and mask needed to decode them are recorded
    /// as structured archive metadata.
    ///
    ///   classified                  display colours supplied by --color-map.
    ///   vector-field:<component-limit>
    ///                               two bands (u, v) for a particle layer.
    ///                               Signed-normalised into red and green with
    ///                               the on-grid flag in blue. The limit is per
    ///                               component, so a field at the limit in both
    ///                               directions has a magnitude sqrt(2) times
    ///                               larger.
    ///   scalar16:<min>,<max>        one band across red and green, on-grid
    ///                               flag in blue. For a regional model, whose
    ///                               grid stops partway across a tile.
    ///   scalar24:<base>,<interval>  one band across all 24 colour bits. More
    ///                               precise, but no room is left for a mask.
    ///                               -10000,0.1 is Mapbox terrain-RGB exactly.
    ///
    /// One image cannot hold a sequence in a tile. When several forecast times
    /// are selected, the output path must contain {seq} or {valid_time}; one
    /// PMTiles archive is written for each time. {reference_time} is also
    /// replaced, and timestamps use UTC YYYYMMDDHHMMSS.
    #[arg(long, value_name = "SPEC", verbatim_doc_comment)]
    raster: Option<String>,

    /// Fixed display colours, with inclusive lower bounds: 0:#00000000,1:#00aaff.
    /// Accepts #RRGGBB or #RRGGBBAA, or @FILE.json with per-class opacity.
    /// Requires --raster classified. Missing
    /// values and values below the first bound are transparent.
    #[arg(long, value_name = "STOPS", allow_hyphen_values = true)]
    color_map: Option<String>,

    /// Pixels across a --raster tile, 256 or 512. A 256-pixel tile at zoom
    /// z + 1 holds the same pixels as a quarter of a 512-pixel one at z, so
    /// the default maximum zoom is one higher and every zoom is merged as the
    /// larger tile one zoom out would be. [default: 512]
    #[arg(long, value_name = "PIXELS")]
    raster_size: Option<u32>,

    /// Write an index of the archives --raster produced to this path.
    ///
    /// Lists each forecast time with the archive holding it, and the encoding
    /// they share, so that a client can blend between two times without having
    /// to discover the set first. The archive paths are relative to the
    /// manifest.
    #[arg(long, value_name = "PATH")]
    manifest: Option<PathBuf>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().compact().init();
    let cli = parse_cli(std::env::args_os().collect()).unwrap_or_else(|error| error.exit());
    // Outputs are written under a temporary name until complete; an
    // interrupted run removes them rather than leaving them behind.
    ctrlc::set_handler(|| {
        remove_unfinished_outputs();
        std::process::exit(130);
    })
    .context("failed to install the interrupt handler")?;
    convert(
        &cli.input,
        &cli.output,
        &ConvertOptions {
            product: cli.product,
            layer_name_pattern: cli.layer_name_pattern,
            layer_count: cli.layer_count,
            min_zoom: cli.min_zoom,
            max_zoom: cli.max_zoom,
            layer_seq_start: cli.layer_seq_start,
            rename: cli.rename,
            quantize: cli.quantize,
            omit_class: cli.omit_class,
            omit_zero: cli.omit_zero,
            aggregation: cli.aggregation,
            fill: cli.fill,
            missing: cli.missing,
            extent: cli.extent,
            buffer: cli.buffer,
            detail: cli.detail,
            raster: cli.raster,
            color_map: cli.color_map,
            raster_size: cli.raster_size,
            manifest: cli.manifest,
            skip_analysis: cli.skip_analysis,
            min_lead_time: cli.min_lead_time,
        },
    )
}

/// Parses the command line, filling options it leaves out from `--profile`.
fn parse_cli(mut args: Vec<OsString>) -> Result<Cli, clap::Error> {
    let matches = Cli::command().try_get_matches_from(&args)?;
    if let Some(profile) = matches.get_one::<PathBuf>("profile") {
        let defaults = profile_args(profile, &matches)
            .map_err(|message| Cli::command().error(ErrorKind::InvalidValue, message))?;
        // Ahead of the given arguments, which a profile never overrides.
        args.splice(1..1, defaults);
        return Cli::try_parse_from(args);
    }
    Cli::from_arg_matches(&matches)
}

/// Turns a profile into arguments for the options `given` does not set.
fn profile_args(path: &Path, given: &ArgMatches) -> Result<Vec<OsString>, String> {
    let file = path.display();
    let text =
        fs::read_to_string(path).map_err(|error| format!("cannot read profile {file}: {error}"))?;
    let profile: Value =
        serde_json::from_str(&text).map_err(|error| format!("invalid profile {file}: {error}"))?;
    let Value::Object(profile) = profile else {
        return Err(format!("profile {file} must hold one JSON object"));
    };
    let command = Cli::command();
    let mut args = Vec::new();
    for (key, value) in &profile {
        if key == "description" {
            continue;
        }
        let option = command
            .get_arguments()
            .find(|option| option.get_long() == Some(key) && key != "profile")
            .ok_or_else(|| format!("profile {file} has unknown option {key:?}"))?;
        if given.value_source(option.get_id().as_str()) == Some(ValueSource::CommandLine) {
            continue;
        }
        let values = match value {
            Value::Array(values) if matches!(option.get_action(), ArgAction::Append) => {
                values.iter().collect()
            }
            value => vec![value],
        };
        for value in values {
            let value = match value {
                Value::Bool(set) if !option.get_action().takes_values() => {
                    if *set {
                        args.push(format!("--{key}").into());
                    }
                    continue;
                }
                _ if !option.get_action().takes_values() => {
                    return Err(format!(
                        "profile option {key:?} is a flag; give true or false"
                    ));
                }
                Value::String(value) => value.clone(),
                Value::Number(value) => value.to_string(),
                Value::Bool(value) => value.to_string(),
                _ => {
                    return Err(format!(
                        "profile option {key:?} takes a string, number or boolean"
                    ));
                }
            };
            // Joined with '=' so that a value starting with '-' stays a value.
            args.push(format!("--{key}={value}").into());
        }
    }
    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_with(profile: &str, extra: &[&str]) -> Result<Cli, clap::Error> {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "gpv2pmtiles-profile-{}-{}.json",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        fs::write(&path, profile).unwrap();
        let mut args: Vec<OsString> = ["gpv2pmtiles", "in.grib2", "out.pmtiles", "--profile"]
            .map(OsString::from)
            .into();
        args.push(path.clone().into());
        args.extend(extra.iter().map(OsString::from));
        let cli = parse_cli(args);
        fs::remove_file(path).unwrap();
        cli
    }

    #[test]
    fn a_profile_supplies_defaults() {
        let cli = parse_with(
            r#"{
                "description": "hourly rain",
                "product": "hrnowc/intensity",
                "layer-seq-start": 1,
                "max-zoom": 9,
                "omit-zero": true,
                "skip-analysis": false,
                "quantize": ["0:0,1:1", "-1:0"]
            }"#,
            &[],
        )
        .unwrap();
        assert_eq!(cli.product.as_deref(), Some("hrnowc/intensity"));
        assert_eq!(cli.layer_seq_start, 1);
        assert_eq!(cli.max_zoom, Some(9));
        assert!(cli.omit_zero);
        assert!(!cli.skip_analysis);
        assert_eq!(cli.quantize, ["0:0,1:1", "-1:0"]);
        assert_eq!(cli.input, Path::new("in.grib2"));
    }

    #[test]
    fn the_command_line_takes_precedence() {
        let cli = parse_with(
            r#"{"max-zoom": 9, "quantize": ["0:0"], "layer-name-pattern": "rain_{seq}"}"#,
            &["--max-zoom", "5", "--quantize", "1:1"],
        )
        .unwrap();
        assert_eq!(cli.max_zoom, Some(5));
        // A repeated option is replaced, not appended to.
        assert_eq!(cli.quantize, ["1:1"]);
        assert_eq!(cli.layer_name_pattern, "rain_{seq}");
    }

    #[test]
    fn a_wrong_profile_is_a_usage_error() {
        for profile in [
            r#"{"max-zom": 1}"#,
            r#"{"profile": "other.json"}"#,
            r#"{"help": true}"#,
            r#"{"version": true}"#,
            r#"{"max-zoom": [1]}"#,
            r#"{"max-zoom": null}"#,
            r#"{"omit-zero": "yes"}"#,
            r#"{"rain": {"max-zoom": 1}}"#,
            r#"{"max-zoom": "x"}"#,
            "[]",
            "",
            r#"{"max-zoom": 1} {"max-zoom": 2}"#,
        ] {
            let error = parse_with(profile, &[]).map(|_| ()).unwrap_err();
            assert_eq!(error.exit_code(), 2, "{profile}: {error}");
        }
    }

    #[test]
    fn a_missing_profile_is_a_usage_error() {
        let args = [
            "gpv2pmtiles",
            "in.grib2",
            "out.pmtiles",
            "--profile",
            "/nonexistent.json",
        ]
        .map(OsString::from)
        .into();
        assert_eq!(parse_cli(args).map(|_| ()).unwrap_err().exit_code(), 2);
    }
}
