# Apartment Generator: Parameter-Driven Layout with Realistic Variation

## Goals (existing + new)

- **apt_rows / apt_cols** drive layout; **apt_door_prob** drives door density; **seed** drives structural variation (not just translation).
- **New 1:** Rooms should be of **random sizes** (from equally sized to very different).
- **New 2:** Doors can be at **different positions** along walls (not fixed, e.g. not always centered).
- **New 3:** Rooms can be arranged in **random order** (layout not a fixed pattern).
- **New 4:** Output should **resemble real-estate apartment floor plans** (exposé style).

---

## Approach

Keep a **grid-based topology** (rows x cols) so connectivity and implementation stay tractable, but:

1. **Variable room sizes** — Do not divide width/height equally. Sample **random row heights** that sum to `height` and **random column widths** that sum to `width` (e.g. random split points or Dirichlet-style fractions), with minimum size per row/col so no cell collapses. Each cell (room) then has size `(col_width[c], row_height[r])`, so rooms can be very different (e.g. one wide strip, one small square).

2. **Random door positions** — For each wall that has a door, place the **gap at a random position** along that wall (e.g. sample gap center uniformly along the segment, or random offset from one end), subject to keeping the gap fully on the segment. No fixed "center only" rule.

3. **Random room arrangement** — Use the **seed** to:
   - Shuffle or randomize the **order of row heights** and **column widths** (e.g. assign the sampled heights to rows in random order, widths to cols in random order), so the "big" row/column is not always in the same place.
   - Optionally randomize which edge gets a door when multiple edges connect the same regions (spanning tree + extra doors still driven by seed and apt_door_prob).

   So the same (rows, cols, door_prob) can produce different "shapes" (e.g. large room top-left vs bottom-right) depending on seed.

4. **Real-estate floor plan look** — Achieved by combining the above: variable room sizes, non-centered doors, and seed-driven arrangement make layouts feel less like a uniform matrix and more like a typical apartment expose (varied room sizes, natural door positions, irregular but connected plan).

---

## Implementation outline

**File:** `v2dlidar/mapgen.py`, function `_generate_apartment_scenario`.

**Steps:**

1. **Random row heights and column widths (variable room sizes)**  
   - `rows = max(1, spec.apt_rows)`, `cols = max(1, spec.apt_cols)`.  
   - Sample `rows` positive heights that sum to `spec.height` (e.g. random fractions with minimum per row so each row has at least ~0.5–1 unit). Similarly sample `cols` widths that sum to `spec.width`.  
   - Use **seed** (and RNG) so same seed gives same sizes; different seeds give different size distributions.

2. **Random arrangement (room order)**  
   - Shuffle the list of row heights with `rng` and assign to row indices 0..rows-1. Shuffle the list of column widths and assign to col indices 0..cols-1. So the "large" row/col can appear anywhere.  
   - Compute cumulative positions: `y_edges[r]` = sum of heights for rows 0..r-1, `x_edges[c]` = sum of widths for cols 0..c-1. Cell `(r, c)` has x in `[x_edges[c], x_edges[c+1]]`, y in `[y_edges[r], y_edges[r+1]]`.

3. **Internal edges (walls)**  
   - Build the list of internal edges: for each adjacent pair of cells (vertical boundary between (r,c) and (r,c+1), horizontal between (r,c) and (r+1,c)), store the segment geometry. Each edge has a length (for door gap sizing).

4. **Spanning tree (connectivity)**  
   - Union-find over cells; shuffle internal edges with `rng`; build spanning tree. Tree edges are "must have door."

5. **Door placement (apt_door_prob + random position)**  
   - For each internal edge: if in spanning tree, add a door; else add a door with probability `apt_door_prob`.  
   - For each door: **position the gap randomly** along the segment (e.g. sample center uniformly along the wall, then place gap of fixed fraction of segment length around that center, clamped to segment).  
   - Emit one or two segments per wall (wall segments on each side of the gap).

6. **Emit segments**  
   - Outer boundary via `_rect_edges`. For each internal edge: if door, emit two segments (before/after gap); else one full segment.

7. **Return**  
   - Same as today: `(segments, spec, interior, res)`.

---

## Summary

| Requirement | Implementation |
|-------------|----------------|
| Rooms of random sizes | Random row heights and column widths (sum to height/width, min size); cell size = (width[c], height[r]). |
| Doors at different positions | For each door, sample gap center (or offset) randomly along that wall segment. |
| Rooms in random order | Shuffle row heights and column widths before assigning to indices so "big" rows/cols can appear anywhere. |
| Resemble real-estate floor plan | Combination of variable sizes, random door positions, and seed-driven arrangement. |
| apt_rows / apt_cols matter | Define number of rows/cols; grid geometry uses random sizes and order. |
| apt_door_prob matters | Probability of door on non-tree edges. |
| Seed drives variation | Sizes, assignment order, spanning tree, extra doors, and door positions all depend on seed. |

No new spec fields or GUI/CLI changes; existing API remains the same.

---

## Implementation checklist

- [ ] Replace `_generate_apartment_scenario` body in `v2dlidar/mapgen.py` with grid-based logic.
- [ ] Random row heights (sum = height, min per row) and column widths (sum = width, min per col); use RNG seeded by `spec.seed`.
- [ ] Shuffle row heights and column widths with RNG before assigning to indices.
- [ ] Build internal edges from cumulative x_edges / y_edges; each edge has geometry and length.
- [ ] Spanning tree over cells (union-find, shuffle edges, add until connected); tree edges get a door.
- [ ] Non-tree edges: add door with probability `apt_door_prob`.
- [ ] For each door: gap position sampled randomly along segment; gap size e.g. 15% of segment length, clamped; emit two segments (before/after gap) or one full segment if no door.
- [ ] Outer boundary: existing `_rect_edges(spec.width/2, spec.height/2, spec.width, spec.height, 0.0)`.
- [ ] Return `(segments, spec, interior, res)` with `interior = np.ones((H, W))`; no changes to `ScenarioSpec`, GUI, or CLI.
