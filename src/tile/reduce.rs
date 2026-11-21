//! Area-weighted statistics for the blocks a low zoom merges cells into.
//!
//! The aggregations a product recommends ([`Aggregation`]) treat every cell as
//! one sample. That is enough for a maximum, but an average then counts a 1 km
//! cell the same as a 250 m one, and a percentile has no notion of how much of
//! a block a value covers. The statistics here weigh each cell by the base
//! cells it spans, `4^power`, which is its share of the block's area on the
//! grid.
//!
//! The part of a block no cell is stored for is either left out or, with a fill
//! value, counted as that value over the area it covers. Precipitation stores
//! no cell where nothing falls, so a fill of zero makes a percentile or a mean
//! describe the whole block rather than only the part where it rains.

use std::str::FromStr;

use anyhow::{Context, Error, Result, ensure};
use gpv_products::model::Aggregation;

/// How a block's value is computed from its cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reducer {
    /// The product's own aggregation, unweighted, exactly as before.
    Legacy(Aggregation),
    /// The area-weighted mean, rounded half to even once at the end.
    Mean,
    /// The smallest value whose cumulative weight reaches `numerator /
    /// denominator` of the total, with no interpolation between values.
    Percentile(Fraction),
}

/// A quantile as an exact fraction, so that the rank is computed without
/// rounding: `p99.5` is 995/1000.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Fraction {
    pub numerator: u64,
    pub denominator: u64,
}

impl Reducer {
    /// Whether the block statistics of [`super::make_layer`] need every cell
    /// counted once, which the chunk margins otherwise break.
    pub(crate) fn is_weighted(self) -> bool {
        !matches!(self, Self::Legacy(_))
    }

    /// The name recorded in the metadata.
    pub(crate) fn name(self) -> String {
        match self {
            Self::Legacy(Aggregation::Max) => "max".into(),
            Self::Legacy(Aggregation::Min) => "min".into(),
            Self::Legacy(Aggregation::RoughAvg) => "rough-avg".into(),
            Self::Legacy(Aggregation::BitOr) => "bit-or".into(),
            Self::Mean => "mean".into(),
            Self::Percentile(fraction) => {
                let percent = fraction.numerator as f64 * 100.0 / fraction.denominator as f64;
                format!("p{percent}")
            }
        }
    }
}

impl FromStr for Reducer {
    type Err = Error;

    /// Parses `max`, `min`, `rough-avg`, `bit-or`, `mean`, or `p<percent>` such
    /// as `p95` or `p99.5`.
    fn from_str(spec: &str) -> Result<Self> {
        let spec = spec.trim();
        Ok(match spec {
            "max" => Self::Legacy(Aggregation::Max),
            "min" => Self::Legacy(Aggregation::Min),
            "rough-avg" => Self::Legacy(Aggregation::RoughAvg),
            "bit-or" => Self::Legacy(Aggregation::BitOr),
            "mean" => Self::Mean,
            _ => {
                let invalid = || {
                    format!(
                        "--aggregation expects max, min, mean, rough-avg, bit-or or p<percent> \
                         such as p95, but found {spec:?}"
                    )
                };
                let percent = spec.strip_prefix('p').with_context(invalid)?;
                let (whole, fraction) = percent.split_once('.').unwrap_or((percent, ""));
                ensure!(
                    !whole.is_empty()
                        && whole.bytes().all(|b| b.is_ascii_digit())
                        && fraction.bytes().all(|b| b.is_ascii_digit())
                        && fraction.len() <= 6,
                    "{}",
                    invalid()
                );
                let denominator = 100 * 10u64.pow(fraction.len() as u32);
                let numerator = format!("{whole}{fraction}")
                    .parse::<u64>()
                    .with_context(invalid)?;
                ensure!(
                    numerator <= denominator,
                    "--aggregation percentile {percent} is above 100"
                );
                Self::Percentile(Fraction {
                    numerator,
                    denominator,
                })
            }
        })
    }
}

/// One band's cells within a block, ready to be reduced.
///
/// Reused across blocks so that a tile allocates it once.
#[derive(Default)]
pub(crate) struct BlockSamples {
    samples: Vec<(i32, u64)>,
    stored_weight: u64,
}

impl BlockSamples {
    pub(crate) fn clear(&mut self) {
        self.samples.clear();
        self.stored_weight = 0;
    }

    #[inline]
    pub(crate) fn push(&mut self, value: i32, weight: u64) {
        self.samples.push((value, weight));
        self.stored_weight += weight;
    }

    /// The block's value, or `None` when no cell of it holds one.
    ///
    /// The fill is counted only for a block that has a stored value of its
    /// own: a block of nothing but fill would turn the whole area outside the
    /// data into fill, out to the edge of the grid.
    pub(crate) fn reduce(
        &mut self,
        reducer: Reducer,
        capacity: u64,
        fill: Option<i32>,
    ) -> Option<i32> {
        if self.samples.is_empty() {
            return None;
        }
        debug_assert!(
            self.stored_weight <= capacity,
            "cells overlap within a block"
        );
        if let Some(fill) = fill {
            let missing = capacity.saturating_sub(self.stored_weight);
            if missing > 0 {
                self.samples.push((fill, missing));
            }
        }
        let total = self.samples.iter().map(|&(_, weight)| weight).sum::<u64>();
        Some(match reducer {
            Reducer::Mean => weighted_mean(&self.samples, total),
            Reducer::Percentile(fraction) => {
                let rank = ((u128::from(fraction.numerator) * u128::from(total))
                    .div_ceil(u128::from(fraction.denominator)) as u64)
                    .max(1);
                // The extremes need no ordering. They are common: p95 of a
                // block of four or sixteen cells is its maximum.
                if rank == total {
                    self.samples.iter().map(|&(value, _)| value).max().unwrap()
                } else if rank == 1 {
                    self.samples.iter().map(|&(value, _)| value).min().unwrap()
                } else {
                    weighted_select(&mut self.samples, rank)
                }
            }
            Reducer::Legacy(_) => unreachable!("legacy aggregations are reduced elsewhere"),
        })
    }
}

/// The weighted mean of the samples, rounded half to even.
fn weighted_mean(samples: &[(i32, u64)], total: u64) -> i32 {
    let sum = samples
        .iter()
        .map(|&(value, weight)| i128::from(value) * i128::from(weight))
        .sum::<i128>();
    let total = i128::from(total);
    let quotient = sum.div_euclid(total);
    let remainder = sum.rem_euclid(total);
    let rounded = match (2 * remainder).cmp(&total) {
        std::cmp::Ordering::Less => quotient,
        std::cmp::Ordering::Greater => quotient + 1,
        std::cmp::Ordering::Equal => quotient + (quotient & 1),
    };
    // The mean lies between the smallest and largest sample, which are i32s
    // other than the missing marker, so this only guards against a bug.
    rounded.clamp(i128::from(i32::MIN) + 1, i128::from(i32::MAX)) as i32
}

/// Below this many samples a sort is cheaper than partitioning.
const SORT_THRESHOLD: usize = 32;

/// The smallest value whose cumulative weight, in ascending order of value,
/// reaches `rank`.
///
/// A three-way quickselect: precipitation repeats a few values across a block,
/// and gathering the equal ones into the middle keeps it linear where a
/// two-way partition would degrade. It falls back to sorting once the work
/// exceeds a few passes, which bounds the worst case.
pub(crate) fn weighted_select(samples: &mut [(i32, u64)], mut rank: u64) -> i32 {
    debug_assert!(rank >= 1);
    debug_assert!(rank <= samples.iter().map(|&(_, weight)| weight).sum::<u64>());
    let budget = 4 * samples.len();
    let mut work = 0;
    let (mut low, mut high) = (0, samples.len());
    loop {
        let part = &mut samples[low..high];
        if part.len() <= SORT_THRESHOLD || work > budget {
            part.sort_unstable_by_key(|&(value, _)| value);
            let mut cumulative = 0;
            for &(value, weight) in part.iter() {
                cumulative += weight;
                if cumulative >= rank {
                    return value;
                }
            }
            unreachable!("the rank exceeds the total weight");
        }
        work += part.len();

        let pivot = median_of_three(part[0].0, part[part.len() / 2].0, part[part.len() - 1].0);
        // Dutch national flag: [less | equal | unexamined | greater].
        let (mut less, mut index, mut greater) = (0, 0, part.len());
        let (mut less_weight, mut equal_weight) = (0u64, 0u64);
        while index < greater {
            let (value, weight) = part[index];
            if value < pivot {
                part.swap(less, index);
                less_weight += weight;
                less += 1;
                index += 1;
            } else if value > pivot {
                greater -= 1;
                part.swap(index, greater);
            } else {
                equal_weight += weight;
                index += 1;
            }
        }

        if rank <= less_weight {
            high = low + less;
        } else if rank <= less_weight + equal_weight {
            return pivot;
        } else {
            rank -= less_weight + equal_weight;
            low += greater;
        }
    }
}

#[inline]
fn median_of_three(a: i32, b: i32, c: i32) -> i32 {
    a.max(b).min(a.min(b).max(c))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(samples: &[(i32, u64)], rank: u64) -> i32 {
        let mut expanded = samples
            .iter()
            .flat_map(|&(value, weight)| std::iter::repeat_n(value, weight as usize))
            .collect::<Vec<_>>();
        expanded.sort_unstable();
        expanded[rank as usize - 1]
    }

    /// A small deterministic generator, so the cases are the same every run.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }
    }

    #[test]
    fn weighted_select_matches_expanding_the_weights() {
        let mut random = Lcg(7);
        for case in 0..3000 {
            let length = 1 + (random.next() % 200) as usize;
            // Few distinct values exercise the equal partition, many the rest.
            let spread = if case % 2 == 0 { 4 } else { 1000 };
            let samples = (0..length)
                .map(|_| {
                    let value = (random.next() % spread) as i32 - (spread / 2) as i32;
                    let weight = 1u64 << (2 * (random.next() % 3));
                    (value, weight)
                })
                .collect::<Vec<_>>();
            let total = samples.iter().map(|&(_, weight)| weight).sum::<u64>();
            for rank in [1, total / 2 + 1, total.saturating_sub(1).max(1), total] {
                let mut working = samples.clone();
                assert_eq!(
                    weighted_select(&mut working, rank),
                    reference(&samples, rank),
                    "case {case} rank {rank}"
                );
            }
        }
    }

    #[test]
    fn mean_rounds_half_to_even_on_both_sides_of_zero() {
        assert_eq!(weighted_mean(&[(1, 1), (2, 1)], 2), 2);
        assert_eq!(weighted_mean(&[(2, 1), (3, 1)], 2), 2);
        assert_eq!(weighted_mean(&[(-1, 1), (-2, 1)], 2), -2);
        assert_eq!(weighted_mean(&[(-2, 1), (-3, 1)], 2), -2);
        assert_eq!(weighted_mean(&[(1, 1), (2, 2)], 3), 2);
        assert_eq!(weighted_mean(&[(10, 16), (0, 1)], 17), 9);
    }

    #[test]
    fn a_fill_counts_over_the_area_no_cell_covers() {
        let mut samples = BlockSamples::default();
        samples.push(8, 1);
        // Three of four base cells are empty, so a fill of zero holds the rank
        // up to three quarters.
        assert_eq!(samples.reduce(Reducer::Mean, 4, Some(0)), Some(2));
        samples.clear();
        samples.push(8, 1);
        let p75 = Reducer::Percentile(Fraction {
            numerator: 75,
            denominator: 100,
        });
        assert_eq!(samples.reduce(p75, 4, Some(0)), Some(0));
        samples.clear();
        samples.push(8, 1);
        let p76 = Reducer::Percentile(Fraction {
            numerator: 76,
            denominator: 100,
        });
        assert_eq!(samples.reduce(p76, 4, Some(0)), Some(8));
        samples.clear();
        samples.push(8, 1);
        assert_eq!(samples.reduce(p75, 4, None), Some(8));
    }

    #[test]
    fn a_block_of_only_fill_stays_missing() {
        let mut samples = BlockSamples::default();
        assert_eq!(samples.reduce(Reducer::Mean, 4, Some(0)), None);
    }

    #[test]
    fn the_hundredth_percentile_is_the_maximum_when_the_fill_is_not_above_it() {
        let mut samples = BlockSamples::default();
        for value in [-5, -3] {
            samples.push(value, 1);
        }
        let p100 = Reducer::Percentile(Fraction {
            numerator: 1,
            denominator: 1,
        });
        // A fill above every stored value wins the top rank instead.
        assert_eq!(samples.reduce(p100, 4, Some(0)), Some(0));
        samples.clear();
        for value in [-5, -3] {
            samples.push(value, 1);
        }
        assert_eq!(samples.reduce(p100, 4, Some(-10)), Some(-3));
    }

    #[test]
    fn percentiles_parse_into_exact_fractions() {
        assert_eq!(
            "p95".parse::<Reducer>().unwrap(),
            Reducer::Percentile(Fraction {
                numerator: 95,
                denominator: 100
            })
        );
        assert_eq!(
            "p99.5".parse::<Reducer>().unwrap(),
            Reducer::Percentile(Fraction {
                numerator: 995,
                denominator: 1000
            })
        );
        assert_eq!(
            "p0".parse::<Reducer>().unwrap(),
            Reducer::Percentile(Fraction {
                numerator: 0,
                denominator: 100
            })
        );
        assert!("p100.1".parse::<Reducer>().is_err());
        assert!("p".parse::<Reducer>().is_err());
        assert!("p-1".parse::<Reducer>().is_err());
        assert!("median".parse::<Reducer>().is_err());
        assert_eq!("p99.5".parse::<Reducer>().unwrap().name(), "p99.5");
    }
}
