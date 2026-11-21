use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use gpv2pmtiles::{ConvertOptions, convert};

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Cli {
    /// Input GRIB2 file. Gzip-compressed input is detected from the .gz suffix.
    input: PathBuf,

    /// Output PMTiles archive. With --raster and several forecast times, this
    /// must contain {seq} or {valid_time}, such as wind_{valid_time}.pmtiles.
    output: PathBuf,

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

    /// How coarsely MVT grid cells are merged at each zoom, as the log2 of the
    /// resolution a merged block approaches: at 8 a block is about 1/256 of
    /// the tile across, at 9 about 1/512. Give one value for every zoom, or
    /// stops as <ZOOM>:<DETAIL>,... where each holds until the next, such as
    /// 1:9,5:8,9:12. Blocks are a power of two cells across, so the size is
    /// the nearest one to the request. Only values from 8 to 30 are accepted,
    /// so blocks can be made finer than the default but not coarser. Raising
    /// the detail stops short of blocks narrower than one unit of --extent,
    /// which would round away with their value, so the extent limits how fine
    /// a detail takes effect; at the default extent that is about detail 11.
    /// The default blocks are kept as they are even when a small --extent
    /// makes them narrower than a unit. [default: 8]
    #[arg(long, value_name = "SPEC")]
    detail: Option<String>,

    /// Experimental: merge the low zooms into a lattice of 2^detail pixels
    /// across each tile, as --raster does, and write them at an extent of the
    /// same size, one unit per pixel. Zooms whose cells are coarser than the
    /// lattice keep the cells of the grid at --extent.
    #[arg(long)]
    lattice: bool,

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

    /// How the cells merged into a block at a low zoom are combined, instead
    /// of the product's own aggregation: max, min, rough-avg or bit-or, as the
    /// products recommend them, or a statistic that weighs each cell by its
    /// area on the grid:
    ///
    ///   mean (avg)    the grid-area-weighted mean; vector raw values round half to even
    ///   p<percent>    the smallest value covering at least that share of
    ///                 the block, without interpolation, such as p95 or p99.5
    ///
    /// Weighted statistics fill unstored cells with zero for precipitation
    /// and snow, and exclude them otherwise. Override with --missing or --fill.
    #[arg(long, value_name = "METHOD", verbatim_doc_comment)]
    aggregation: Option<String>,

    /// Physical value counted over the part of a block no cell is stored for,
    /// with --aggregation mean or p<percent>, overriding --missing for a band.
    /// Precipitation and snow already default to zero fill. The value must be one
    /// the band can hold exactly. Prefix with a band name when the product has
    /// several bands, and repeat as needed. A block with no stored cell stays
    /// empty.
    #[arg(long, value_name = "VALUE")]
    fill: Vec<String>,

    /// Weighted statistics: exclude missing cells or fill with a physical value.
    /// Defaults to zero for precipitation/snow, exclusion for other products.
    #[arg(long, value_name = "POLICY", allow_hyphen_values = true)]
    missing: Option<String>,

    /// Write RGBA raster tiles instead of MVT, encoding band values into the
    /// colour channels so that a shader can read them back.
    ///
    /// The parameters belong to the archive rather than to a tile, so that
    /// every tile decodes the same way and the field does not jump at a seam.
    /// The channels, offset, scale and mask needed to decode them are recorded
    /// as structured archive metadata.
    ///
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
    let cli = Cli::parse();
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
            lattice: cli.lattice,
            raster: cli.raster,
            raster_size: cli.raster_size,
            manifest: cli.manifest,
            skip_analysis: cli.skip_analysis,
            min_lead_time: cli.min_lead_time,
        },
    )
}
