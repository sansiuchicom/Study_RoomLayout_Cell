"""Per-part axis-aligned atomization.

Each territory piece is rotated into the owning part's local frame, covered by
a grid whose lines pass through piece vertices (so reflex/clip corners are atom
boundaries) and through ``DimensionPolicy``-sized subdivisions between them.
Cells are clipped against the (local-frame) piece polygon, then rotated back to
global coordinates.

Curved parts use only their bounding box as anchors (no per-vertex anchors,
since dense curve vertices would explode the grid) and are atomized in the
global frame (effective theta = 0).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import degrees

import shapely.affinity as sa
import shapely.geometry as sg
from shapely.geometry.polygon import orient as _orient
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .dimensions import DimensionPolicy, snap_length, split_interval
from .schema import ShapeInput, ShapePart
from .structural_guides import build_structural_guides
from .territory import KIND_CURVED, resolve_territories


@dataclass(frozen=True)
class Atom:
    atom_id: int
    shape: ShapePart
    part_id: int
    piece_id: int
    theta: float
    is_feature_sliver: bool

    @property
    def area(self) -> float:
        return float(_to_shapely(self.shape).area)

    @property
    def centroid(self) -> tuple[float, float]:
        c = _to_shapely(self.shape).centroid
        return float(c.x), float(c.y)


def atomize(
    shape: ShapeInput,
    policy: DimensionPolicy | None = None,
    *,
    absorb_slivers: bool = True,
) -> tuple[Atom, ...]:
    """Atomize all territories with anchor-sharing within theta groups.

    Pieces with the same effective theta share a local frame, so they share
    grid anchors — atoms from different parts in the same theta group align at
    their boundaries. Pieces in different theta groups (e.g. axis-aligned main
    vs rotated wing) cannot share a grid; each group is gridded independently.
    """
    policy = policy or DimensionPolicy()
    territories = resolve_territories(shape)
    structural_guides = build_structural_guides(shape, territories)

    # Group pieces by their effective theta (curved is treated as 0).
    groups: dict[float, list[tuple]] = defaultdict(list)
    for terr in territories:
        is_curved = terr.kind == KIND_CURVED
        eff_theta = 0.0 if is_curved else terr.theta
        key = round(eff_theta, 9)
        for piece_idx, piece in enumerate(terr.pieces):
            groups[key].append((eff_theta, is_curved, terr.part_id, piece_idx, piece))

    atoms: list[Atom] = []
    next_id = [0]
    for key, members in groups.items():
        if not members:
            continue
        eff_theta = members[0][0]

        # Rotate every piece in this group into the shared local frame.
        local_pieces = []
        for _, is_curved, part_id, piece_idx, piece in members:
            piece_poly = _to_shapely(piece)
            if piece_poly.is_empty or piece_poly.area < 1e-9:
                continue
            if abs(eff_theta) > 1e-12:
                local_poly = sa.rotate(piece_poly, -degrees(eff_theta), origin=(0, 0))
            else:
                local_poly = piece_poly
            local_pieces.append((local_poly, is_curved, part_id, piece_idx))

        if not local_pieces:
            continue

        # Gather shared anchors from every piece in the group.
        all_xs: set[float] = set()
        all_ys: set[float] = set()
        for local_poly, is_curved, _, _ in local_pieces:
            xs, ys = _piece_anchors(local_poly, is_curved)
            all_xs.update(xs)
            all_ys.update(ys)
        guide = structural_guides.get(key)
        if guide is not None:
            all_xs.update(guide.xs)
            all_ys.update(guide.ys)
        if not all_xs or not all_ys:
            continue

        x_grid = _expand_anchors(sorted(all_xs), policy)
        y_grid = _expand_anchors(sorted(all_ys), policy)

        for local_poly, _, part_id, piece_idx in local_pieces:
            atoms.extend(
                _atomize_with_shared_grid(
                    local_poly=local_poly,
                    x_grid=x_grid,
                    y_grid=y_grid,
                    theta=eff_theta,
                    part_id=part_id,
                    piece_idx=piece_idx,
                    next_id=next_id,
                    policy=policy,
                )
            )

    if absorb_slivers:
        atoms = _absorb_slivers(atoms, policy)
    return tuple(atoms)


def _absorb_slivers(atoms: list[Atom], policy: DimensionPolicy) -> list[Atom]:
    """Merge sliver atoms into their largest-shared-boundary neighbor.

    Smaller slivers first. Each sliver is unioned with the neighbor it shares
    the longest boundary with; the neighbor keeps its metadata (atom_id,
    part_id, piece_id, theta). Orphan slivers with no neighbor are left as-is.
    """
    threshold = policy.min_atom_size * policy.min_atom_size * 0.5
    atom_map: dict[int, Atom] = {a.atom_id: a for a in atoms}
    poly_map: dict[int, sg.Polygon] = {
        a.atom_id: _to_shapely(a.shape) for a in atoms
    }

    max_passes = 8
    for _ in range(max_passes):
        slivers = sorted(
            (aid for aid in atom_map if poly_map[aid].area < threshold),
            key=lambda aid: poly_map[aid].area,
        )
        if not slivers:
            break

        ids_list = list(atom_map.keys())
        polys_list = [poly_map[aid] for aid in ids_list]
        tree = STRtree(polys_list)

        absorbed: set[int] = set()
        merged_any = False

        for sliver_id in slivers:
            if sliver_id in absorbed:
                continue
            sliver_poly = poly_map[sliver_id]

            best_target = None
            best_length = 0.0
            for j in tree.query(sliver_poly):
                j = int(j)
                other_id = ids_list[j]
                if other_id == sliver_id or other_id in absorbed:
                    continue
                if other_id not in atom_map:
                    continue
                shared = sliver_poly.intersection(poly_map[other_id])
                length = _line_length(shared)
                if length > best_length:
                    best_length = length
                    best_target = other_id

            if best_target is None:
                continue

            merged = unary_union([sliver_poly, poly_map[best_target]])
            if not isinstance(merged, sg.Polygon) or merged.is_empty:
                continue

            poly_map[best_target] = merged
            target = atom_map[best_target]
            atom_map[best_target] = Atom(
                atom_id=target.atom_id,
                shape=_from_shapely(merged),
                part_id=target.part_id,
                piece_id=target.piece_id,
                theta=target.theta,
                is_feature_sliver=merged.area < threshold,
            )
            del atom_map[sliver_id]
            del poly_map[sliver_id]
            absorbed.add(sliver_id)
            merged_any = True

        if not merged_any:
            break

    return list(atom_map.values())


def _line_length(geom) -> float:
    if geom.is_empty:
        return 0.0
    if geom.geom_type == "LineString":
        return float(geom.length)
    if geom.geom_type == "MultiLineString":
        return sum(float(g.length) for g in geom.geoms)
    if geom.geom_type == "GeometryCollection":
        total = 0.0
        for g in geom.geoms:
            total += _line_length(g)
        return total
    return 0.0


def _atomize_with_shared_grid(
    local_poly: sg.Polygon,
    x_grid: list[float],
    y_grid: list[float],
    theta: float,
    part_id: int,
    piece_idx: int,
    next_id: list[int],
    policy: DimensionPolicy,
) -> list[Atom]:
    minx, miny, maxx, maxy = local_poly.bounds
    sliver_area = policy.min_atom_size * policy.min_atom_size * 0.5

    # Restrict the grid sweep to this piece's local bbox; cells outside it
    # cannot intersect the piece, so we save the intersection cost.
    x_lo, x_hi = _grid_window(x_grid, minx, maxx)
    y_lo, y_hi = _grid_window(y_grid, miny, maxy)

    atoms: list[Atom] = []
    for ix in range(x_lo, x_hi):
        x0, x1 = x_grid[ix], x_grid[ix + 1]
        if x1 - x0 <= 1e-9:
            continue
        for iy in range(y_lo, y_hi):
            y0, y1 = y_grid[iy], y_grid[iy + 1]
            if y1 - y0 <= 1e-9:
                continue
            cell = sg.box(x0, y0, x1, y1)
            clipped = local_poly.intersection(cell)
            if clipped.is_empty or clipped.area < 1e-9:
                continue
            for sub in _polygon_parts(clipped):
                if abs(theta) > 1e-12:
                    global_poly = sa.rotate(sub, degrees(theta), origin=(0, 0))
                else:
                    global_poly = sub
                atoms.append(
                    Atom(
                        atom_id=next_id[0],
                        shape=_from_shapely(global_poly),
                        part_id=part_id,
                        piece_id=piece_idx,
                        theta=theta,
                        is_feature_sliver=global_poly.area < sliver_area,
                    )
                )
                next_id[0] += 1

    return atoms


def _grid_window(grid: list[float], lo: float, hi: float) -> tuple[int, int]:
    """Return [start, end) indices into ``grid`` whose cells overlap [lo, hi]."""
    n_cells = len(grid) - 1
    start = 0
    while start < n_cells and grid[start + 1] <= lo + 1e-9:
        start += 1
    end = n_cells
    while end > start and grid[end - 1] >= hi - 1e-9:
        end -= 1
    return start, end


def _piece_anchors(local_poly: sg.Polygon, is_curved: bool):
    minx, miny, maxx, maxy = local_poly.bounds
    if is_curved:
        return [minx, maxx], [miny, maxy]

    xs = {round(minx, 6), round(maxx, 6)}
    ys = {round(miny, 6), round(maxy, 6)}
    for ring in [local_poly.exterior, *local_poly.interiors]:
        for x, y in list(ring.coords):
            xs.add(round(x, 6))
            ys.add(round(y, 6))
    return sorted(xs), sorted(ys)


def _expand_anchors(anchors: list[float], policy: DimensionPolicy) -> list[float]:
    """Place grid lines between successive anchors via ``split_interval``.

    Anchors come from polygon vertices in the local frame and need not be on
    the geometry-snap grid (rotated rectangles produce irrational coordinates).
    For each gap, snap the length to the geometry grid before calling
    ``split_interval`` (which strictly requires snap-aligned inputs), then place
    the resulting widths from the unsnapped anchor. The boundary anchor is
    always appended last so the polygon edge stays put; the last cell absorbs
    any drift between snapped vs raw gap (<= geometry_snap / 2 ≈ 5mm).
    """
    if len(anchors) < 2:
        return list(anchors)
    snap = policy.geometry_snap
    out = [anchors[0]]
    for i in range(len(anchors) - 1):
        a, b = anchors[i], anchors[i + 1]
        raw_gap = b - a
        if raw_gap <= 1e-9:
            continue
        if raw_gap <= policy.min_atom_size + 1e-9:
            if b > out[-1] + 1e-9:
                out.append(b)
            continue
        gap_snapped = snap_length(raw_gap, snap)
        widths = split_interval(gap_snapped, policy)
        pos = a
        for w in widths[:-1]:
            pos = pos + w
            if pos > out[-1] + 1e-9 and pos < b - 1e-9:
                out.append(pos)
        if b > out[-1] + 1e-9:
            out.append(b)
    return out


def _to_shapely(part: ShapePart) -> sg.Polygon:
    return sg.Polygon(part.exterior, [list(h) for h in part.holes])


def _from_shapely(poly: sg.Polygon) -> ShapePart:
    poly = _orient(poly, sign=1.0)
    ext = tuple(tuple(map(float, p)) for p in list(poly.exterior.coords)[:-1])
    holes = tuple(
        tuple(tuple(map(float, p)) for p in list(ring.coords)[:-1])
        for ring in poly.interiors
    )
    return ShapePart(exterior=ext, holes=holes)


def _polygon_parts(geom) -> list:
    if geom.is_empty:
        return []
    if isinstance(geom, sg.Polygon):
        return [geom]
    if isinstance(geom, sg.MultiPolygon):
        return [p for p in geom.geoms if isinstance(p, sg.Polygon) and not p.is_empty]
    if hasattr(geom, "geoms"):
        out = []
        for part in geom.geoms:
            out.extend(_polygon_parts(part))
        return out
    return []
