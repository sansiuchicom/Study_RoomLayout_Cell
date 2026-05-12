"""Structural guide extraction shared by atomizer and regionizer.

Synthetic cases already know their design-time primitives. This module keeps
the important vertex/seam coordinates as first-class data so atom and region
boundaries can make the same alignment decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import degrees, hypot

import shapely.affinity as sa
import shapely.geometry as sg
from shapely.ops import unary_union

from .schema import ShapeInput, ShapePart
from .territory import KIND_CURVED, Territory, resolve_territories


ROUND_DECIMALS = 6
STRAIGHT_SEGMENT_MIN = 0.5
AXIS_TOL = 1e-6


@dataclass(frozen=True)
class StructuralGuide:
    """Local-frame structural coordinates for one effective-theta group."""

    theta: float
    xs: tuple[float, ...]
    ys: tuple[float, ...]
    points: tuple[tuple[float, float], ...]
    event_xs: tuple[float, ...]
    event_ys: tuple[float, ...]
    event_points: tuple[tuple[float, float], ...]
    guard_xs: tuple[float, ...]
    guard_ys: tuple[float, ...]
    guard_points: tuple[tuple[float, float], ...]


def build_structural_guides(
    shape: ShapeInput,
    territories: tuple[Territory, ...] | None = None,
) -> dict[float, StructuralGuide]:
    """Return structural guides keyed by rounded effective theta.

    Straight-edged territory pieces contribute all vertices. Curved pieces
    contribute only sparse event vertices: bbox extrema and endpoints of long
    straight clip/seam segments. This preserves curved-straight transition
    points such as ``(4, 8)`` in case 28 without turning every arc segment into
    a hard grid line.
    """

    if territories is None:
        territories = resolve_territories(shape)
    groups: dict[float, dict[str, set]] = {}

    for terr in territories:
        eff_theta = 0.0 if terr.kind == KIND_CURVED else terr.theta
        key = round(eff_theta, 9)
        group = groups.setdefault(
            key,
            {
                "theta": eff_theta,
                "xs": set(),
                "ys": set(),
                "points": set(),
                "event_xs": set(),
                "event_ys": set(),
                "event_points": set(),
                "guard_xs": set(),
                "guard_ys": set(),
                "guard_points": set(),
                "straight_polys": [],
            },
        )

        for piece in terr.pieces:
            poly = _to_shapely(piece)
            if poly.is_empty or poly.area < 1e-9:
                continue
            local_poly = _rotate_geom(poly, -eff_theta)
            if terr.kind == KIND_CURVED:
                points = _curved_event_points(local_poly)
                event_points = points
            else:
                points = _all_ring_points(local_poly)
                event_points = set()
                group["straight_polys"].append(local_poly)

            for x, y in points:
                point = (_snap(x), _snap(y))
                group["points"].add(point)
                group["xs"].add(point[0])
                group["ys"].add(point[1])
            for x, y in event_points:
                point = (_snap(x), _snap(y))
                group["event_points"].add(point)
                group["event_xs"].add(point[0])
                group["event_ys"].add(point[1])
                group["guard_points"].add(point)
                group["guard_xs"].add(point[0])
                group["guard_ys"].add(point[1])

    for group in groups.values():
        if group["straight_polys"]:
            for x, y in _reflex_points(unary_union(group["straight_polys"])):
                point = (_snap(x), _snap(y))
                if point in group["points"]:
                    group["guard_points"].add(point)
                    group["guard_xs"].add(point[0])
                    group["guard_ys"].add(point[1])

    return {
        key: StructuralGuide(
            theta=float(group["theta"]),
            xs=tuple(sorted(group["xs"])),
            ys=tuple(sorted(group["ys"])),
            points=tuple(sorted(group["points"])),
            event_xs=tuple(sorted(group["event_xs"])),
            event_ys=tuple(sorted(group["event_ys"])),
            event_points=tuple(sorted(group["event_points"])),
            guard_xs=tuple(sorted(group["guard_xs"])),
            guard_ys=tuple(sorted(group["guard_ys"])),
            guard_points=tuple(sorted(group["guard_points"])),
        )
        for key, group in groups.items()
    }


def _curved_event_points(poly: sg.Polygon) -> set[tuple[float, float]]:
    minx, miny, maxx, maxy = poly.bounds
    out: set[tuple[float, float]] = set()

    for ring in [poly.exterior, *poly.interiors]:
        coords = list(ring.coords)
        if len(coords) < 2:
            continue
        open_coords = coords[:-1]

        for x, y in open_coords:
            if (
                abs(x - minx) < AXIS_TOL
                or abs(x - maxx) < AXIS_TOL
                or abs(y - miny) < AXIS_TOL
                or abs(y - maxy) < AXIS_TOL
            ):
                out.add((float(x), float(y)))

        for a, b in zip(coords, coords[1:]):
            x0, y0 = a
            x1, y1 = b
            if hypot(x1 - x0, y1 - y0) < STRAIGHT_SEGMENT_MIN:
                continue
            if abs(x1 - x0) < AXIS_TOL or abs(y1 - y0) < AXIS_TOL:
                out.add((float(x0), float(y0)))
                out.add((float(x1), float(y1)))

    return out


def _all_ring_points(poly: sg.Polygon) -> set[tuple[float, float]]:
    out: set[tuple[float, float]] = set()
    for ring in [poly.exterior, *poly.interiors]:
        for x, y in list(ring.coords)[:-1]:
            out.add((float(x), float(y)))
    return out


def _reflex_points(geom) -> set[tuple[float, float]]:
    out: set[tuple[float, float]] = set()
    for poly in _polygon_parts(geom):
        if not poly.exterior.is_ccw:
            poly = sg.Polygon(
                list(poly.exterior.coords)[::-1],
                [list(h.coords)[::-1] for h in poly.interiors],
            )
        _scan_reflex_ring(list(poly.exterior.coords)[:-1], out)
        for hole in poly.interiors:
            coords = list(hole.coords)[:-1]
            _scan_reflex_ring(coords[::-1] if hole.is_ccw else coords, out)
    return out


def _scan_reflex_ring(coords, out: set[tuple[float, float]]) -> None:
    n = len(coords)
    for i in range(n):
        ax, ay = coords[(i - 1) % n]
        bx, by = coords[i]
        cx, cy = coords[(i + 1) % n]
        v1x, v1y = bx - ax, by - ay
        v2x, v2y = cx - bx, cy - by
        if v1x * v2y - v1y * v2x < -AXIS_TOL:
            out.add((float(bx), float(by)))


def _polygon_parts(geom) -> list[sg.Polygon]:
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


def _rotate_geom(geom, theta_rad):
    if abs(theta_rad) < 1e-12:
        return geom
    return sa.rotate(geom, degrees(theta_rad), origin=(0, 0))


def _to_shapely(part: ShapePart) -> sg.Polygon:
    return sg.Polygon(part.exterior, [list(h) for h in part.holes])


def _snap(value: float) -> float:
    return round(float(value), ROUND_DECIMALS)
