"""Atom-based regionizer with cross-piece merging.

Algorithm:

1. Group atoms by effective theta (curved → 0, otherwise atom.theta).
2. Within each theta group, find connected components via the atom graph
   (cross-piece edges merge axis-aligned parts that share a boundary).
3. For each component, union atom polygons into one merged piece.
4. Apply hierarchical polygon-cut partition on the merged piece. Cut
   candidates are filtered so they align with atom column/row positions,
   mirroring the reference algorithm but ensuring every cut coincides with
   atom edges (no thin region-boundary slivers).
5. Assign atoms to leaf sub-pieces by local-frame centroid containment.

The merging step is what makes region boundaries line up across the (formerly
independent) part-piece boundaries — analogous to the atom-phase fix where
same-theta pieces shared anchors.

Cut hierarchy (from reference at algorithm/celllayout/zoning.py):
    T1a cross_cut       — V + H at a polygon vertex (atom-aligned)
    T1b vertex_aligned  — V or H at a polygon vertex (atom-aligned axis)
    T2  reflex_pair     — oblique line between two reflex vertices (atom
                          alignment not possible; region boundary follows
                          atom edges via centroid assignment)
    T3  axis_mid        — V or H at an atom anchor inside [0.3, 0.7] bbox
                          fraction (atom-aligned)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import cos, degrees, sin

import numpy as np
import shapely.affinity as sa
import shapely.geometry as sg
from shapely.geometry.polygon import orient as _orient
from shapely.ops import split, unary_union

from .atom_graph import AtomGraph, build_atom_graph
from .atomize import Atom, atomize
from .dimensions import DimensionPolicy
from .schema import ShapeInput, ShapePart
from .structural_guides import build_structural_guides
from .territory import KIND_CURVED, resolve_territories


# Reference parameters --------------------------------------------------------
MIN_AREA = 3.0
FEATURE_MIN_AREA = 0.5
MARGIN = 0.5
MIN_CUT_LEN = 1.0
MAX_ASPECT = 4.0
BAL_MIN = 0.15
FEATURE_BAL_MIN = 0.0
TIE_DECIMALS = 6


@dataclass(frozen=True)
class Region:
    region_id: int
    shape: ShapePart
    atom_ids: tuple[int, ...]
    part_ids: tuple[int, ...]
    piece_keys: tuple[tuple[int, int], ...]
    theta: float
    cut_history: tuple[str, ...]


@dataclass(frozen=True)
class _PartitionContext:
    structural: dict
    atom_xs: tuple[float, ...]
    atom_ys: tuple[float, ...]
    atom_xs_set: frozenset[float]
    atom_ys_set: frozenset[float]
    hard_xs: tuple[float, ...]
    hard_ys: tuple[float, ...]
    hard_points: tuple[tuple[float, float], ...]


def regionize(
    shape: ShapeInput,
    atoms: tuple[Atom, ...] | None = None,
    atom_graph: AtomGraph | None = None,
    policy: DimensionPolicy | None = None,
    *,
    target_area: float = 6.0,
) -> tuple[Region, ...]:
    if atoms is None:
        atoms = atomize(shape, policy)
    if not atoms:
        return ()
    atoms_list = list(atoms)
    if atom_graph is None:
        atom_graph = build_atom_graph(shape, atoms=tuple(atoms_list))

    territories = resolve_territories(shape)
    structural_guides = build_structural_guides(shape, territories)
    terr_by_part = {t.part_id: t for t in territories}

    def _eff_theta(atom: Atom) -> float:
        terr = terr_by_part.get(atom.part_id)
        is_curved = (terr.kind == KIND_CURVED) if terr else False
        return 0.0 if is_curved else atom.theta

    # Group atom indices by effective theta
    theta_groups: dict[float, list[int]] = defaultdict(list)
    for idx, a in enumerate(atoms_list):
        key = round(_eff_theta(a), 9)
        theta_groups[key].append(idx)

    # Build adjacency by atom INDEX (matches atom_graph edge encoding)
    adjacency: dict[int, list[int]] = defaultdict(list)
    for e in atom_graph.edges:
        adjacency[e.atom_a].append(e.atom_b)
        adjacency[e.atom_b].append(e.atom_a)

    regions: list[Region] = []
    next_id = [0]

    for group_indices in theta_groups.values():
        group_set = set(group_indices)
        for comp_indices in _components_within(group_set, adjacency):
            comp_atoms = [atoms_list[i] for i in comp_indices]
            atom_polys = [_to_shapely(a.shape) for a in comp_atoms]
            merged_poly = unary_union(atom_polys)
            if isinstance(merged_poly, sg.MultiPolygon):
                merged_poly = max(merged_poly.geoms, key=lambda p: p.area)
            if not isinstance(merged_poly, sg.Polygon) or merged_poly.area < 1e-9:
                continue

            eff_theta = _eff_theta(comp_atoms[0])
            local_poly = _rotate_geom(merged_poly, -eff_theta)
            atoms_with_local = [
                (a, _rotate_point(a.centroid, -eff_theta)) for a in comp_atoms
            ]
            local_atom_polys = [_rotate_geom(p, -eff_theta) for p in atom_polys]

            atom_xs = _collect_atom_edge_positions(local_atom_polys, axis="x")
            atom_ys = _collect_atom_edge_positions(local_atom_polys, axis="y")
            atom_xs_set = frozenset(round(x, 6) for x in atom_xs)
            atom_ys_set = frozenset(round(y, 6) for y in atom_ys)
            guide = structural_guides.get(round(eff_theta, 9))
            hard_xs = (
                tuple(x for x in guide.event_xs if round(x, 6) in atom_xs_set)
                if guide is not None else ()
            )
            hard_ys = (
                tuple(y for y in guide.event_ys if round(y, 6) in atom_ys_set)
                if guide is not None else ()
            )
            hard_points = (
                tuple(
                    (x, y) for x, y in guide.event_points
                    if round(x, 6) in atom_xs_set and round(y, 6) in atom_ys_set
                )
                if guide is not None else ()
            )
            ctx = _PartitionContext(
                structural=_structural_coords(local_poly),
                atom_xs=atom_xs,
                atom_ys=atom_ys,
                atom_xs_set=atom_xs_set,
                atom_ys_set=atom_ys_set,
                hard_xs=hard_xs,
                hard_ys=hard_ys,
                hard_points=hard_points,
            )

            k = max(1, round(merged_poly.area / target_area))
            groups = _recurse_partition(local_poly, atoms_with_local, k, ctx)

            for atom_list, cut_history in groups:
                actual_atoms = [aw[0] for aw in atom_list]
                if not actual_atoms:
                    continue
                shape_part = _union_atoms_to_shape_part(actual_atoms)
                if shape_part is None:
                    continue
                part_ids = tuple(sorted({a.part_id for a in actual_atoms}))
                piece_keys = tuple(sorted(
                    {(a.part_id, a.piece_id) for a in actual_atoms}
                ))
                regions.append(
                    Region(
                        region_id=next_id[0],
                        shape=shape_part,
                        atom_ids=tuple(a.atom_id for a in actual_atoms),
                        part_ids=part_ids,
                        piece_keys=piece_keys,
                        theta=eff_theta,
                        cut_history=tuple(cut_history),
                    )
                )
                next_id[0] += 1

    return tuple(regions)


def _components_within(group_set, adjacency):
    unvisited = set(group_set)
    components = []
    while unvisited:
        start = next(iter(unvisited))
        stack = [start]
        comp = []
        unvisited.discard(start)
        while stack:
            node = stack.pop()
            comp.append(node)
            for nb in adjacency.get(node, ()):
                if nb in unvisited:
                    unvisited.discard(nb)
                    stack.append(nb)
        components.append(comp)
    return components


# Recursive partition ---------------------------------------------------------


def _recurse_partition(local_poly, atoms_with_local, k, ctx):
    if k <= 1 or local_poly.area < MIN_AREA * 2:
        sel = _select_feature_cut(local_poly, ctx)
        if sel is not None:
            return _recurse_from_cut(sel, atoms_with_local, max(k, 1), ctx)
        return [(atoms_with_local, [])]

    sel = _select_cut(local_poly, k, ctx)
    if sel is None:
        return [(atoms_with_local, [])]

    return _recurse_from_cut(sel, atoms_with_local, k, ctx)


def _recurse_from_cut(sel, atoms_with_local, k, ctx):
    label, _lines, pieces, _b = sel
    sub_atoms_lists: list[list] = [[] for _ in pieces]
    for aw in atoms_with_local:
        pt = sg.Point(aw[1])
        assigned = False
        for i, sub_poly in enumerate(pieces):
            if sub_poly.contains(pt):
                sub_atoms_lists[i].append(aw)
                assigned = True
                break
        if not assigned:
            best_i = min(
                range(len(pieces)),
                key=lambda i: pieces[i].distance(pt),
            )
            sub_atoms_lists[best_i].append(aw)

    result = []
    for sub_poly, sub_atoms, sub_k in zip(
        pieces, sub_atoms_lists, _allocate_k(pieces, k),
    ):
        for group_atoms, group_history in _recurse_partition(
            sub_poly, sub_atoms, sub_k, ctx,
        ):
            result.append((group_atoms, [label] + group_history))
    return result


# Cut selection ---------------------------------------------------------------


def _select_cut(local_poly, k_total, ctx):
    for label, gen, prefer_short, bmin, min_area in (
        ("structural_cross",
         lambda: _structural_cross_cut_pairs(local_poly, ctx),
         False, BAL_MIN, MIN_AREA),
        ("structural_axis",
         lambda: ([ln] for ln in _structural_axis_lines(local_poly, ctx)),
         False, BAL_MIN, MIN_AREA),
        ("cross_cut", lambda: _cross_cut_pairs(local_poly, ctx), False, BAL_MIN, MIN_AREA),
        ("vertex_aligned",
         lambda: ([ln] for ln in _vertex_aligned_lines(local_poly, ctx)),
         False, BAL_MIN, MIN_AREA),
        ("reflex_pair", lambda: _reflex_pair_lines(local_poly), True, BAL_MIN, MIN_AREA),
        ("axis_mid",
         lambda: _axis_mid_lines_atom_aligned(local_poly, ctx),
         False, 0.0, MIN_AREA),
    ):
        cands = ((label, lines) for lines in gen())
        r = _best_cut(cands, local_poly, bmin, k_total, prefer_short, min_area)
        if r is not None:
            return r
    return None


def _select_feature_cut(local_poly, ctx):
    """Try hard structural cuts even when target-area recursion has stopped."""
    for label, gen, prefer_short in (
        ("structural_cross",
         lambda: _structural_cross_cut_pairs(local_poly, ctx),
         False),
        ("structural_axis",
         lambda: ([ln] for ln in _structural_axis_lines(local_poly, ctx)),
         False),
    ):
        cands = ((label, lines) for lines in gen())
        r = _best_cut(
            cands,
            local_poly,
            FEATURE_BAL_MIN,
            None,
            prefer_short,
            FEATURE_MIN_AREA,
        )
        if r is not None:
            return r
    return None


def _vertex_coords_raw(poly):
    coords = list(poly.exterior.coords)[:-1]
    for h in poly.interiors:
        coords.extend(list(h.coords)[:-1])
    return coords


def _reflex_vertices(poly):
    if not poly.exterior.is_ccw:
        poly = sg.Polygon(
            list(poly.exterior.coords)[::-1],
            [list(h.coords)[::-1] for h in poly.interiors],
        )
    out = []

    def scan(coords):
        n = len(coords)
        for i in range(n):
            a, b, c = (np.asarray(coords[(i + j - 1) % n]) for j in range(3))
            v1, v2 = b - a, c - b
            if v1[0] * v2[1] - v1[1] * v2[0] < -1e-6:
                out.append(tuple(b))

    scan(list(poly.exterior.coords)[:-1])
    for h in poly.interiors:
        c = list(h.coords)[:-1]
        scan(c[::-1] if h.is_ccw else c)
    return out


def _structural_coords(poly):
    rfx = _reflex_vertices(poly)
    return {
        "xs": {round(x, 6) for x, _ in rfx},
        "ys": {round(y, 6) for _, y in rfx},
    }


def _structural_axis_lines(poly, ctx):
    minx, miny, maxx, maxy = poly.bounds
    cuts, sx, sy = [], set(), set()
    for x in ctx.hard_xs:
        kx = round(x, 2)
        if minx + MARGIN < x < maxx - MARGIN and kx not in sx:
            sx.add(kx)
            cuts.append(sg.LineString([(x, miny - 1), (x, maxy + 1)]))
    for y in ctx.hard_ys:
        ky = round(y, 2)
        if miny + MARGIN < y < maxy - MARGIN and ky not in sy:
            sy.add(ky)
            cuts.append(sg.LineString([(minx - 1, y), (maxx + 1, y)]))
    return cuts


def _structural_cross_cut_pairs(poly, ctx):
    minx, miny, maxx, maxy = poly.bounds
    pairs, seen = [], set()
    for x, y in ctx.hard_points:
        k = (round(x, 2), round(y, 2))
        if (
            k in seen
            or round(x, 6) not in ctx.atom_xs_set
            or round(y, 6) not in ctx.atom_ys_set
            or not (minx + MARGIN < x < maxx - MARGIN)
            or not (miny + MARGIN < y < maxy - MARGIN)
        ):
            continue
        seen.add(k)
        pairs.append(
            [
                sg.LineString([(x, miny - 1), (x, maxy + 1)]),
                sg.LineString([(minx - 1, y), (maxx + 1, y)]),
            ]
        )
    return pairs


def _vertex_aligned_lines(poly, ctx):
    """T1b: V/H per polygon vertex, restricted to atom-anchor positions.

    Parent reflex coords carried via ``ctx.structural`` are also tried, but
    only if they appear in the atom anchor set (typically true for axis-
    aligned shapes since atom anchors include every polygon vertex).
    """
    minx, miny, maxx, maxy = poly.bounds
    coords = _vertex_coords_raw(poly)
    if ctx.structural:
        coords += [(x, miny) for x in ctx.structural["xs"]]
        coords += [(minx, y) for y in ctx.structural["ys"]]

    cuts, sx, sy = [], set(), set()
    for x, y in coords:
        kx, ky = round(x, 2), round(y, 2)
        x_aligned = round(x, 6) in ctx.atom_xs_set
        y_aligned = round(y, 6) in ctx.atom_ys_set
        if x_aligned and minx + MARGIN < x < maxx - MARGIN and kx not in sx:
            sx.add(kx)
            cuts.append(sg.LineString([(x, miny - 1), (x, maxy + 1)]))
        if y_aligned and miny + MARGIN < y < maxy - MARGIN and ky not in sy:
            sy.add(ky)
            cuts.append(sg.LineString([(minx - 1, y), (maxx + 1, y)]))
    return cuts


def _cross_cut_pairs(poly, ctx):
    """T1a: V+H pair per polygon vertex, both axes must align with atom anchors."""
    minx, miny, maxx, maxy = poly.bounds
    pairs, seen = [], set()
    for x, y in _vertex_coords_raw(poly):
        if (
            round(x, 6) not in ctx.atom_xs_set
            or round(y, 6) not in ctx.atom_ys_set
        ):
            continue
        k = (round(x, 2), round(y, 2))
        if (
            k in seen
            or not (minx + MARGIN < x < maxx - MARGIN)
            or not (miny + MARGIN < y < maxy - MARGIN)
        ):
            continue
        seen.add(k)
        pairs.append(
            [
                sg.LineString([(x, miny - 1), (x, maxy + 1)]),
                sg.LineString([(minx - 1, y), (maxx + 1, y)]),
            ]
        )
    return pairs


def _reflex_pair_lines(poly):
    rfx = _reflex_vertices(poly)
    out = []
    for i in range(len(rfx)):
        for j in range(i + 1, len(rfx)):
            line = sg.LineString([rfx[i], rfx[j]])
            inter = line.intersection(poly)
            if not (hasattr(inter, "length") and inter.length >= MIN_CUT_LEN):
                continue
            (x1, y1), (x2, y2) = list(line.coords)[0], list(line.coords)[-1]
            if abs(x1 - x2) < 1e-3 or abs(y1 - y2) < 1e-3:
                continue
            out.append([line])
    return out


def _axis_mid_lines_atom_aligned(poly, ctx):
    minx, miny, maxx, maxy = poly.bounds
    W = maxx - minx
    H = maxy - miny
    if W <= 0 or H <= 0:
        return []
    x_lo, x_hi = minx + 0.3 * W, minx + 0.7 * W
    y_lo, y_hi = miny + 0.3 * H, miny + 0.7 * H

    cuts = []
    for x in ctx.atom_xs:
        if x_lo <= x <= x_hi:
            cuts.append([sg.LineString([(x, miny - 1), (x, maxy + 1)])])
    for y in ctx.atom_ys:
        if y_lo <= y <= y_hi:
            cuts.append([sg.LineString([(minx - 1, y), (maxx + 1, y)])])
    return cuts


# Split / validity ------------------------------------------------------------


def _split_pieces(poly, lines):
    pieces = [poly]
    for line in lines:
        nxt = []
        for p in pieces:
            try:
                r = split(p, line)
                parts = list(r.geoms) if hasattr(r, "geoms") else [r]
                nxt.extend(
                    q for q in parts if isinstance(q, sg.Polygon) and q.area > 0.01
                )
            except Exception:
                nxt.append(p)
        pieces = nxt or pieces
    if len(pieces) < 2:
        return None
    return sorted(pieces, key=lambda p: -p.area)


def _piece_aspect(p):
    if p.is_empty or p.area < 1e-6:
        return 99.0
    try:
        c = list(p.minimum_rotated_rectangle.exterior.coords)
        e1 = float(np.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1]))
        e2 = float(np.hypot(c[2][0] - c[1][0], c[2][1] - c[1][1]))
        return max(e1, e2) / max(min(e1, e2), 1e-6)
    except Exception:
        return 99.0


def _balance(pieces):
    a = [p.area for p in pieces]
    return min(a) / max(a)


def _allocate_k(pieces, k_total):
    total = sum(p.area for p in pieces)
    out, acc = [], 0
    for i, p in enumerate(pieces):
        if i == len(pieces) - 1:
            kk = max(1, k_total - acc)
        else:
            kk = max(1, round(k_total * p.area / total))
            acc += kk
        out.append(kk)
    return out


def _aspect_ok(pieces, k_total):
    if k_total is None:
        return all(_piece_aspect(p) <= MAX_ASPECT for p in pieces)
    for p, kp in zip(pieces, _allocate_k(pieces, k_total)):
        if (kp <= 1 or p.area < MIN_AREA * 2) and _piece_aspect(p) > MAX_ASPECT:
            return False
    return True


def _best_cut(
    candidates,
    poly,
    bal_min,
    k_total,
    prefer_short=False,
    min_area=MIN_AREA,
):
    valid = []
    for label, lines in candidates:
        pieces = _split_pieces(poly, lines)
        if pieces is None or min(p.area for p in pieces) < min_area:
            continue
        b = _balance(pieces)
        if b < bal_min or not _aspect_ok(pieces, k_total):
            continue
        valid.append((label, lines, pieces, b))
    if not valid:
        return None
    if prefer_short:
        valid.sort(
            key=lambda v: (
                -round(v[3], TIE_DECIMALS),
                sum(line.length for line in v[1]),
            )
        )
    else:
        valid.sort(
            key=lambda v: (
                -round(v[3], TIE_DECIMALS),
                max(_piece_aspect(p) for p in v[2]),
            )
        )
    return valid[0]


# Geometry helpers ------------------------------------------------------------


def _to_shapely(part: ShapePart) -> sg.Polygon:
    return sg.Polygon(part.exterior, [list(h) for h in part.holes])


def _rotate_geom(geom, theta_rad):
    if abs(theta_rad) < 1e-12:
        return geom
    return sa.rotate(geom, degrees(theta_rad), origin=(0, 0))


def _rotate_point(pt, theta_rad):
    if abs(theta_rad) < 1e-12:
        return pt
    c, s = cos(theta_rad), sin(theta_rad)
    x, y = pt
    return (x * c - y * s, x * s + y * c)


def _collect_atom_edge_positions(local_polys, axis="x"):
    positions: set[float] = set()
    for poly in local_polys:
        if poly.is_empty:
            continue
        for x, y in list(poly.exterior.coords)[:-1]:
            positions.add(round(x if axis == "x" else y, 6))
    return tuple(sorted(positions))


def _union_atoms_to_shape_part(atoms) -> ShapePart | None:
    polys = [_to_shapely(a.shape) for a in atoms]
    if not polys:
        return None
    merged = unary_union(polys)
    if merged.is_empty:
        return None
    if isinstance(merged, sg.MultiPolygon):
        merged = max(merged.geoms, key=lambda p: p.area)
    if not isinstance(merged, sg.Polygon):
        return None
    merged = _orient(merged, sign=1.0)
    ext = tuple(tuple(map(float, p)) for p in list(merged.exterior.coords)[:-1])
    holes = tuple(
        tuple(tuple(map(float, p)) for p in list(r.coords)[:-1])
        for r in merged.interiors
    )
    return ShapePart(exterior=ext, holes=holes)
