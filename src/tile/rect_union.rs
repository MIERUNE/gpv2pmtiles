use anyhow::{Result, bail, ensure};
use fast_mvt::{MvtCoord, MvtLineString, MvtPolygon};
use foldhash::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct IntRect {
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
}

impl IntRect {
    pub(super) fn new(x1: i32, y1: i32, x2: i32, y2: i32) -> Self {
        debug_assert!(x1 < x2 && y1 < y2);
        Self { x1, y1, x2, y2 }
    }
}

#[derive(Clone, Copy, Debug)]
struct Edge {
    start: MvtCoord,
    end: MvtCoord,
    direction: u8,
}

impl Edge {
    fn new(start: MvtCoord, end: MvtCoord) -> Self {
        debug_assert!(start != end);
        let direction = if start.y == end.y {
            if start.x < end.x { 0 } else { 2 }
        } else if start.y < end.y {
            1
        } else {
            3
        };
        Self {
            start,
            end,
            direction,
        }
    }
}

/// Coordinates are clamped to `[-buffer, extent + buffer]`, both bounded by
/// `MAX_MVT_BUFFER * MAX_MVT_EXTENT / 256`, so 21 bits hold one with an offset.
const COORD_OFFSET: i64 = 1 << 20;
const COORD_BITS: u32 = 21;
const COORD_MASK: i64 = (1 << COORD_BITS) - 1;

#[inline]
fn coord_code(value: i32) -> u64 {
    debug_assert!((i64::from(value) + COORD_OFFSET) & !COORD_MASK == 0);
    ((i64::from(value) + COORD_OFFSET) & COORD_MASK) as u64
}

/// The order `merge_horizontal_runs` sorts in, as one comparable integer.
#[inline]
fn run_sort_key(rectangle: IntRect) -> u64 {
    (coord_code(rectangle.y1) << (2 * COORD_BITS))
        | (coord_code(rectangle.y2) << COORD_BITS)
        | coord_code(rectangle.x1)
}

/// The order the sweep ends bands in, as one comparable integer.
#[inline]
fn end_sort_key(band: IntRect, index: u32) -> u64 {
    (coord_code(band.y2) << 32) | u64::from(index)
}

/// Unions axis-aligned rectangles and returns MVT-ready polygons.
///
/// A horizontal sweep maintains the covered X intervals above and below each
/// event line. Their difference forms horizontal boundaries; the ends of the
/// covered intervals form vertical boundaries between event lines.
///
/// Event lines where the coverage is unchanged are skipped, so a run of rows
/// with the same horizontal extent produces one tall band rather than one band
/// per row. Raster input is full of such runs, and every skipped band is two
/// edges per covered interval that the contour tracer never has to stitch back
/// together.
pub(super) fn union_rectangles(mut rectangles: Vec<IntRect>) -> Result<Vec<MvtPolygon>> {
    if rectangles.is_empty() {
        return Ok(Vec::new());
    }
    if rectangles.len() == 1 {
        return Ok(vec![rectangle_polygon(rectangles[0])]);
    }

    // Rows of neighbouring cells collapse into single rectangles here, which
    // shrinks both the sweep's event count and its active set.
    merge_horizontal_runs(&mut rectangles);
    if rectangles.len() == 1 {
        return Ok(vec![rectangle_polygon(rectangles[0])]);
    }
    let bands = rectangles;
    let mut by_end = bands
        .iter()
        .enumerate()
        .map(|(index, band)| end_sort_key(*band, index as u32))
        .collect::<Vec<_>>();
    by_end.sort_unstable();

    // Sorted by (x1, x2), so the covered intervals fall out of a linear scan
    // and never have to be re-sorted.
    let mut active = Vec::<(i32, i32, u32)>::new();
    let mut covered_above = Vec::<(i32, i32)>::new();
    let mut covered_below = Vec::<(i32, i32)>::new();
    // A band contributes a pair of vertical edges and its two horizontal ends.
    let mut edges = Vec::<Edge>::with_capacity(bands.len().saturating_mul(4));
    let mut band_start_y = 0;

    let (mut start_index, mut end_index) = (0, 0);
    while start_index < bands.len() || end_index < by_end.len() {
        let y = match (
            bands.get(start_index),
            by_end
                .get(end_index)
                .map(|&entry| bands[entry as u32 as usize]),
        ) {
            (Some(starting), Some(ending)) => starting.y1.min(ending.y2),
            (Some(starting), None) => starting.y1,
            (None, Some(ending)) => ending.y2,
            (None, None) => unreachable!("the loop condition keeps one side non-empty"),
        };

        while let Some(band) = bands.get(start_index).filter(|band| band.y1 == y) {
            add_interval(&mut active, band.x1, band.x2);
            start_index += 1;
        }
        while let Some(&entry) = by_end.get(end_index) {
            let band = bands[entry as u32 as usize];
            if band.y2 != y {
                break;
            }
            remove_interval(&mut active, band.x1, band.x2);
            end_index += 1;
        }

        merge_active(&active, &mut covered_below);
        if covered_below == covered_above {
            continue;
        }

        for &(x1, x2) in &covered_above {
            // Clockwise outer boundaries and counter-clockwise holes keep the
            // filled area on the right side of every directed edge.
            edges.push(Edge::new(
                MvtCoord { x: x1, y },
                MvtCoord {
                    x: x1,
                    y: band_start_y,
                },
            ));
            edges.push(Edge::new(
                MvtCoord {
                    x: x2,
                    y: band_start_y,
                },
                MvtCoord { x: x2, y },
            ));
        }
        add_horizontal_edges(y, &covered_above, &covered_below, &mut edges);
        std::mem::swap(&mut covered_above, &mut covered_below);
        band_start_y = y;
    }

    contours_to_polygons(trace_contours(&edges)?)
}

#[inline]
fn rectangle_polygon(rectangle: IntRect) -> MvtPolygon {
    MvtPolygon::new(
        MvtLineString::new(vec![
            MvtCoord {
                x: rectangle.x1,
                y: rectangle.y1,
            },
            MvtCoord {
                x: rectangle.x2,
                y: rectangle.y1,
            },
            MvtCoord {
                x: rectangle.x2,
                y: rectangle.y2,
            },
            MvtCoord {
                x: rectangle.x1,
                y: rectangle.y2,
            },
        ]),
        Vec::new(),
    )
}

/// Sorts the rectangles by row and merges the ones that share a row and touch,
/// so that each row is left with disjoint intervals in ascending order.
///
/// Compact coordinate ranges use integer sort keys; only surviving runs are
/// decoded. Wide coordinates retain the general in-place comparison sort.
fn merge_horizontal_runs(rectangles: &mut Vec<IntRect>) {
    // Four coordinates fit in a u64 when their span needs at most 16 bits.
    // Keep the general comparator for wide extents and coordinate ranges.
    let min = rectangles.iter().map(|r| r.x1.min(r.y1)).min().unwrap_or(0);
    let max = rectangles
        .iter()
        .map(|r| r.x2.max(r.y2))
        .max()
        .unwrap_or(min);
    let span = (i64::from(max) - i64::from(min)) as u64;
    let bits = (u64::BITS - span.leading_zeros()).max(1);
    if bits <= 16 {
        merge_packed_runs(rectangles, min, bits);
        return;
    }
    rectangles.sort_unstable_by_key(|rectangle| run_sort_key(*rectangle));

    let mut kept = 0usize;
    for read in 0..rectangles.len() {
        let rectangle = rectangles[read];
        let last = kept.checked_sub(1).map(|index| rectangles[index]);
        if let Some(last) = last
            && last.y1 == rectangle.y1
            && last.y2 == rectangle.y2
            && rectangle.x1 <= last.x2
        {
            rectangles[kept - 1].x2 = last.x2.max(rectangle.x2);
        } else {
            rectangles[kept] = rectangle;
            kept += 1;
        }
    }
    rectangles.truncate(kept);
}

/// Sort compact integer keys, then decode only the surviving horizontal runs.
fn merge_packed_runs(rectangles: &mut Vec<IntRect>, offset: i32, bits: u32) {
    let mask = (1u64 << bits) - 1;
    let coordinate = |value: i32| (i64::from(value) - i64::from(offset)) as u64;
    let mut packed = rectangles
        .iter()
        .map(|r| {
            coordinate(r.y1) << (3 * bits)
                | coordinate(r.y2) << (2 * bits)
                | coordinate(r.x1) << bits
                | coordinate(r.x2)
        })
        .collect::<Vec<_>>();
    packed.sort_unstable();
    let mut kept = 0;
    for index in 0..packed.len() {
        let key = packed[index];
        if kept > 0
            && packed[kept - 1] >> (2 * bits) == key >> (2 * bits)
            && (key >> bits & mask) <= (packed[kept - 1] & mask)
        {
            packed[kept - 1] =
                (packed[kept - 1] & !mask) | (packed[kept - 1] & mask).max(key & mask);
        } else {
            packed[kept] = key;
            kept += 1;
        }
    }
    let decode = |key: u64, shift| (i64::from(offset) + ((key >> shift) & mask) as i64) as i32;
    for index in 0..kept {
        let key = packed[index];
        rectangles[index] = IntRect::new(
            decode(key, bits),
            decode(key, 3 * bits),
            decode(key, 0),
            decode(key, 2 * bits),
        );
    }
    rectangles.truncate(kept);
}

#[inline]
fn add_interval(active: &mut Vec<(i32, i32, u32)>, x1: i32, x2: i32) {
    // The sweep opens bands in ascending (x1, x2) order within a line, so the
    // common case appends to the end. Both others keep the sorted invariant.
    if let Some(last) = active.last_mut() {
        if (x1, x2) == (last.0, last.1) {
            last.2 += 1;
            return;
        }
        if (x1, x2) > (last.0, last.1) {
            active.push((x1, x2, 1));
            return;
        }
    } else {
        active.push((x1, x2, 1));
        return;
    }
    match active.binary_search_by(|probe| (probe.0, probe.1).cmp(&(x1, x2))) {
        Ok(index) => active[index].2 += 1,
        Err(index) => active.insert(index, (x1, x2, 1)),
    }
}

#[inline]
fn remove_interval(active: &mut Vec<(i32, i32, u32)>, x1: i32, x2: i32) {
    if let Some(last) = active.last_mut()
        && (last.0, last.1) == (x1, x2)
    {
        last.2 -= 1;
        if last.2 == 0 {
            active.pop();
        }
        return;
    }
    let Ok(index) = active.binary_search_by(|probe| (probe.0, probe.1).cmp(&(x1, x2))) else {
        debug_assert!(false, "every end event matches a start event");
        return;
    };
    active[index].2 -= 1;
    if active[index].2 == 0 {
        active.remove(index);
    }
}

/// Collapses the sorted active intervals into disjoint covered intervals.
fn merge_active(active: &[(i32, i32, u32)], covered: &mut Vec<(i32, i32)>) {
    covered.clear();
    for &(x1, x2, _) in active {
        if let Some(last) = covered.last_mut()
            && x1 <= last.1
        {
            last.1 = last.1.max(x2);
        } else {
            covered.push((x1, x2));
        }
    }
}

/// Merges two sorted endpoint streams; coverage toggles at each endpoint.
/// This emits the same segments in the same order as sorting all endpoints.
fn add_horizontal_edges(y: i32, above: &[(i32, i32)], below: &[(i32, i32)], edges: &mut Vec<Edge>) {
    let mut above = above.iter().flat_map(|&(x1, x2)| [x1, x2]).peekable();
    let mut below = below.iter().flat_map(|&(x1, x2)| [x1, x2]).peekable();
    let (mut is_above, mut is_below) = (false, false);
    while let Some(x1) = first_endpoint(above.peek(), below.peek()) {
        while above.next_if_eq(&x1).is_some() {
            is_above = !is_above;
        }
        while below.next_if_eq(&x1).is_some() {
            is_below = !is_below;
        }
        let Some(x2) = first_endpoint(above.peek(), below.peek()) else {
            break;
        };
        match (is_above, is_below) {
            (false, true) => edges.push(Edge::new(MvtCoord { x: x1, y }, MvtCoord { x: x2, y })),
            (true, false) => edges.push(Edge::new(MvtCoord { x: x2, y }, MvtCoord { x: x1, y })),
            _ => {}
        }
    }
}

#[inline]
fn first_endpoint(left: Option<&i32>, right: Option<&i32>) -> Option<i32> {
    match (left, right) {
        (Some(&left), Some(&right)) => Some(left.min(right)),
        (Some(&value), None) | (None, Some(&value)) => Some(value),
        (None, None) => None,
    }
}

/// The ends of a vertex's outgoing-edge chain, in edge order.
///
/// The chain links two consecutive edges through `next`, so one insertion per
/// edge builds it - no second lookup to place the edge in a flat array, which
/// the tracer used to pay for every edge of every boundary.
struct VertexChain {
    head: u32,
    tail: u32,
}

const NO_EDGE: u32 = u32::MAX;

fn trace_contours(edges: &[Edge]) -> Result<Vec<MvtLineString>> {
    ensure!(
        edges.len() <= NO_EDGE as usize,
        "rectangle union produced too many edges at once"
    );
    // The tracer needs the outgoing edges of a vertex twice per edge - once to
    // record it and once to follow it - and most vertices have exactly one.
    // A `Vec` per vertex used to be a heap allocation per edge; a chain of
    // index links is one map entry per vertex and one array slot per edge.
    let mut vertices =
        HashMap::<(i32, i32), VertexChain>::with_capacity_and_hasher(edges.len(), <_>::default());
    let mut next = vec![NO_EDGE; edges.len()];
    for (index, edge) in edges.iter().enumerate() {
        let index = index as u32;
        match vertices.entry((edge.start.x, edge.start.y)) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                let chain = entry.get_mut();
                next[chain.tail as usize] = index;
                chain.tail = index;
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(VertexChain {
                    head: index,
                    tail: index,
                });
            }
        }
    }

    let mut visited = vec![false; edges.len()];
    let mut contours = Vec::new();
    for first_index in 0..edges.len() {
        if visited[first_index] {
            continue;
        }
        let first = edges[first_index];
        // Rings are short, so one allocation up front covers almost all of them.
        let mut points = Vec::with_capacity(8);
        points.push(first.start);
        let mut edge_index = first_index;
        loop {
            ensure!(
                !visited[edge_index],
                "rectangle union boundary loops before closing"
            );
            visited[edge_index] = true;
            let edge = edges[edge_index];
            if edge.end == first.start {
                break;
            }

            let chain = vertices
                .get(&(edge.end.x, edge.end.y))
                .ok_or_else(|| anyhow::anyhow!("rectangle union produced an open boundary"))?;
            let mut next_index = NO_EDGE;
            let mut best_priority = u8::MAX;
            let mut candidate = chain.head;
            while candidate != NO_EDGE {
                if !visited[candidate as usize] {
                    let priority =
                        turn_priority(edge.direction, edges[candidate as usize].direction);
                    if priority < best_priority {
                        best_priority = priority;
                        next_index = candidate;
                    }
                }
                candidate = next[candidate as usize];
            }
            if next_index == NO_EDGE {
                return Err(anyhow::anyhow!("rectangle union produced an open boundary"));
            }
            let next_index = next_index as usize;
            if edges[next_index].direction != edge.direction {
                points.push(edge.end);
            }
            edge_index = next_index;
        }
        ensure!(
            points.len() >= 4,
            "rectangle union produced a degenerate ring"
        );
        contours.push(MvtLineString::new(points));
    }
    Ok(contours)
}

#[inline]
fn turn_priority(incoming: u8, outgoing: u8) -> u8 {
    match (outgoing + 4 - incoming) % 4 {
        1 => 0, // right
        0 => 1, // straight
        3 => 2, // left
        2 => 3, // back
        _ => unreachable!(),
    }
}

/// An exterior ring while its holes are still being assigned to it.
struct Ring {
    /// Doubled signed area, positive because the ring is an exterior one.
    area: i64,
    /// `[x1, y1, x2, y2]`.
    bounds: [i32; 4],
    exterior: MvtLineString,
    interiors: Vec<MvtLineString>,
}

impl Ring {
    /// The bounds reject nearly every ring for the price of four comparisons,
    /// which keeps the ray casting off the hot path.
    fn contains(&self, point: MvtCoord) -> bool {
        self.bounds[0] <= point.x
            && point.x <= self.bounds[2]
            && self.bounds[1] <= point.y
            && point.y <= self.bounds[3]
            && contains(&self.exterior, point)
    }
}

fn contours_to_polygons(contours: Vec<MvtLineString>) -> Result<Vec<MvtPolygon>> {
    let mut polygons = Vec::<Ring>::new();
    let mut holes = Vec::new();
    for contour in contours {
        let (area, bounds) = area_and_bounds(&contour);
        ensure!(area != 0, "rectangle union produced a zero-area ring");
        if area > 0 {
            polygons.push(Ring {
                area,
                bounds,
                exterior: contour,
                interiors: Vec::new(),
            });
        } else {
            holes.push(contour);
        }
    }

    for hole in holes {
        let sample = hole.0[0];
        let Some(ring) = polygons
            .iter_mut()
            .filter(|ring| ring.contains(sample))
            .min_by_key(|ring| ring.area)
        else {
            bail!("rectangle union produced a hole without an exterior ring");
        };
        ring.interiors.push(hole);
    }

    Ok(polygons
        .into_iter()
        .map(|ring| MvtPolygon::new(ring.exterior, ring.interiors))
        .collect())
}

/// The doubled signed area and the `[x1, y1, x2, y2]` bounds of a ring, in
/// one pass: every point is read two or three times otherwise.
fn area_and_bounds(ring: &MvtLineString) -> (i64, [i32; 4]) {
    let mut bounds = [i32::MAX, i32::MAX, i32::MIN, i32::MIN];
    let mut area = 0i64;
    let mut previous = ring.0[ring.0.len() - 1];
    for point in &ring.0 {
        area +=
            i64::from(previous.x) * i64::from(point.y) - i64::from(previous.y) * i64::from(point.x);
        previous = *point;
        bounds[0] = bounds[0].min(point.x);
        bounds[1] = bounds[1].min(point.y);
        bounds[2] = bounds[2].max(point.x);
        bounds[3] = bounds[3].max(point.y);
    }
    (area, bounds)
}

/// Casts a ray to the left of `point`. Every ring here is rectilinear, so only
/// the vertical edges can be crossed and the crossing needs no interpolation.
fn contains(ring: &MvtLineString, point: MvtCoord) -> bool {
    let mut inside = false;
    for (index, &left) in ring.0.iter().enumerate() {
        let right = ring.0[(index + 1) % ring.0.len()];
        if left.x == right.x && (left.y > point.y) != (right.y > point.y) && point.x < left.x {
            inside = !inside;
        }
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_run_sort_matches_the_general_sort_at_signed_and_bit_boundaries() {
        for offset in [i32::MIN, -65536, -32, 0, i32::MAX - 65535] {
            let mut rectangles = vec![
                IntRect::new(offset + 1, offset, offset + 3, offset + 2),
                IntRect::new(offset, offset, offset + 1, offset + 2),
                IntRect::new(offset + 3, offset, offset + 5, offset + 2),
                IntRect::new(offset, offset, offset + 4, offset + 2),
                IntRect::new(offset, offset + 2, offset + 65535, offset + 65535),
            ];
            let mut expected = rectangles.clone();
            expected.sort_unstable_by_key(|r| {
                run_sort_key(IntRect::new(
                    r.x1 - offset,
                    r.y1 - offset,
                    r.x2 - offset,
                    r.y2 - offset,
                ))
            });
            let mut result = Vec::<IntRect>::new();
            for r in expected {
                if let Some(last) = result.last_mut()
                    && last.y1 == r.y1
                    && last.y2 == r.y2
                    && r.x1 <= last.x2
                {
                    last.x2 = last.x2.max(r.x2);
                } else {
                    result.push(r);
                }
            }
            merge_horizontal_runs(&mut rectangles);
            assert_eq!(rectangles, result);
        }
        let mut wide = vec![
            IntRect::new(0, 0, 40000, 1),
            IntRect::new(40000, 0, 70000, 1),
        ];
        merge_horizontal_runs(&mut wide);
        assert_eq!(wide, [IntRect::new(0, 0, 70000, 1)]);
    }

    #[test]
    fn horizontal_edges_merge_overlapping_and_touching_intervals_in_order() {
        let mut edges = Vec::new();
        add_horizontal_edges(3, &[(0, 2), (4, 8)], &[(1, 5), (7, 9)], &mut edges);
        let coordinates = edges
            .iter()
            .map(|edge| (edge.start.x, edge.start.y, edge.end.x, edge.end.y))
            .collect::<Vec<_>>();
        assert_eq!(
            coordinates,
            [(1, 3, 0, 3), (2, 3, 4, 3), (7, 3, 5, 3), (8, 3, 9, 3)]
        );

        edges.clear();
        add_horizontal_edges(0, &[(0, 2)], &[(2, 4)], &mut edges);
        assert_eq!(edges.len(), 2);
        assert_eq!((edges[0].start.x, edges[0].end.x), (2, 0));
        assert_eq!((edges[1].start.x, edges[1].end.x), (2, 4));
    }

    #[test]
    fn horizontal_edges_handle_empty_equal_and_extreme_coordinates() {
        let mut edges = Vec::new();
        add_horizontal_edges(0, &[], &[(i32::MIN, -1), (0, i32::MAX)], &mut edges);
        assert_eq!(edges.len(), 2);
        assert_eq!((edges[0].start.x, edges[0].end.x), (i32::MIN, -1));
        assert_eq!((edges[1].start.x, edges[1].end.x), (0, i32::MAX));
        edges.clear();
        add_horizontal_edges(0, &[(1, 2)], &[(1, 2)], &mut edges);
        add_horizontal_edges(0, &[], &[], &mut edges);
        assert!(edges.is_empty());
    }

    fn area(polygons: &[MvtPolygon]) -> i64 {
        polygons
            .iter()
            .map(|polygon| {
                area_and_bounds(polygon.exterior()).0
                    + polygon
                        .interiors()
                        .iter()
                        .map(|ring| area_and_bounds(ring).0)
                        .sum::<i64>()
            })
            .sum::<i64>()
            / 2
    }

    #[test]
    fn merges_adjacent_and_overlapping_rectangles() {
        let polygons = union_rectangles(vec![
            IntRect::new(0, 0, 2, 2),
            IntRect::new(2, 0, 4, 1),
            IntRect::new(1, 1, 3, 3),
        ])
        .unwrap();

        assert_eq!(polygons.len(), 1);
        assert!(polygons[0].interiors().is_empty());
        assert_eq!(area(&polygons), 9);
    }

    #[test]
    fn removes_duplicates_and_contained_rectangles() {
        let polygons = union_rectangles(vec![
            IntRect::new(0, 0, 4, 4),
            IntRect::new(0, 0, 4, 4),
            IntRect::new(1, 1, 3, 3),
        ])
        .unwrap();

        assert_eq!(polygons.len(), 1);
        assert_eq!(polygons[0].exterior().0.len(), 5);
        assert_eq!(area(&polygons), 16);
    }

    #[test]
    fn assigns_a_hole_to_its_exterior() {
        let polygons = union_rectangles(vec![
            IntRect::new(0, 0, 4, 1),
            IntRect::new(0, 3, 4, 4),
            IntRect::new(0, 1, 1, 3),
            IntRect::new(3, 1, 4, 3),
        ])
        .unwrap();

        assert_eq!(polygons.len(), 1);
        assert_eq!(polygons[0].interiors().len(), 1);
        assert_eq!(area(&polygons), 12);
    }

    #[test]
    fn keeps_corner_touching_rectangles_as_separate_polygons() {
        let polygons =
            union_rectangles(vec![IntRect::new(0, 0, 1, 1), IntRect::new(1, 1, 2, 2)]).unwrap();

        assert_eq!(polygons.len(), 2);
        assert_eq!(area(&polygons), 2);
    }

    #[test]
    fn keeps_an_island_inside_a_hole() {
        let polygons = union_rectangles(vec![
            IntRect::new(0, 0, 5, 1),
            IntRect::new(0, 4, 5, 5),
            IntRect::new(0, 1, 1, 4),
            IntRect::new(4, 1, 5, 4),
            IntRect::new(2, 2, 3, 3),
        ])
        .unwrap();

        assert_eq!(polygons.len(), 2);
        assert_eq!(area(&polygons), 17);
    }

    /// Rasterizing both the input and the output compares the sweep against an
    /// oracle that shares none of its machinery. The larger grids matter as
    /// much as the dense ones: they are what produce the long runs of
    /// unchanged rows and the multi-ring boundaries that the sweep takes its
    /// shortcuts on.
    #[test]
    fn randomized_rectangles_preserve_every_covered_cell() {
        let mut state = 0x6a09_e667_f3bc_c909_u64;
        for size in [4, 8, 16, 24] {
            for case in 0..400 {
                let count = (next_random(&mut state) % 40 + 1) as usize;
                let rectangles = (0..count)
                    .map(|_| {
                        let x1 = (next_random(&mut state) % size as u64) as i32;
                        let y1 = (next_random(&mut state) % size as u64) as i32;
                        let x2 = x1 + 1 + (next_random(&mut state) % (size - x1) as u64) as i32;
                        let y2 = y1 + 1 + (next_random(&mut state) % (size - y1) as u64) as i32;
                        IntRect::new(x1, y1, x2, y2)
                    })
                    .collect::<Vec<_>>();
                let polygons = union_rectangles(rectangles.clone()).unwrap();

                let mut covered = 0;
                for y in 0..size {
                    for x in 0..size {
                        let expected = rectangles.iter().any(|rectangle| {
                            rectangle.x1 <= x
                                && x < rectangle.x2
                                && rectangle.y1 <= y
                                && y < rectangle.y2
                        });
                        let actual = polygons.iter().any(|polygon| {
                            contains_point(polygon.exterior(), x as f64 + 0.5, y as f64 + 0.5)
                                && !polygon.interiors().iter().any(|hole| {
                                    contains_point(hole, x as f64 + 0.5, y as f64 + 0.5)
                                })
                        });
                        assert_eq!(
                            actual, expected,
                            "size {size}, case {case}, cell ({x}, {y})"
                        );
                        covered += i64::from(expected);
                    }
                }

                // The rings also have to enclose that area exactly, which
                // catches a hole that was attached to the wrong exterior.
                assert_eq!(area(&polygons), covered, "size {size}, case {case}");
            }
        }
    }

    fn next_random(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    fn contains_point(ring: &MvtLineString, x: f64, y: f64) -> bool {
        let mut inside = false;
        for (left, right) in ring
            .0
            .iter()
            .zip(ring.0.iter().cycle().skip(1))
            .take(ring.0.len())
        {
            if (f64::from(left.y) > y) != (f64::from(right.y) > y) {
                let intersection_x = f64::from(right.x - left.x) * (y - f64::from(left.y))
                    / f64::from(right.y - left.y)
                    + f64::from(left.x);
                if x < intersection_x {
                    inside = !inside;
                }
            }
        }
        inside
    }

    #[test]
    fn an_empty_input_produces_no_polygons() {
        assert!(union_rectangles(Vec::new()).unwrap().is_empty());
    }
}
