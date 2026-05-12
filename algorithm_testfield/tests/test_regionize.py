import math

import shapely.affinity as sa
import shapely.geometry as sg
from shapely.ops import unary_union

from celllayout_tf.atomize import atomize
from celllayout_tf.cases import selected_cases
from celllayout_tf.regionize import Region, regionize
from celllayout_tf.structural_guides import build_structural_guides


def _shape_to_polygon(shape):
    return sg.Polygon(shape.exterior, [list(h) for h in shape.holes])


def _local_bounds(shape, theta):
    poly = _shape_to_polygon(shape)
    if abs(theta) > 1e-12:
        poly = sa.rotate(poly, -math.degrees(theta), origin=(0, 0))
    return poly.bounds


def _total_atom_area(atoms):
    return sum(_shape_to_polygon(a.shape).area for a in atoms)


def test_regionize_rect_returns_multiple_regions():
    case = selected_cases([1])[0][2]  # 30평 판상형 14×10 = 140 m²
    regions = regionize(case)
    assert len(regions) >= 5  # 140 / 6 ≈ 23


def test_every_atom_assigned_to_exactly_one_region():
    case = selected_cases([1])[0][2]
    atoms = atomize(case)
    regions = regionize(case, atoms=atoms)

    all_atom_ids = {a.atom_id for a in atoms}
    assigned: set[int] = set()
    for r in regions:
        for aid in r.atom_ids:
            assert aid not in assigned, f"atom {aid} assigned twice"
            assigned.add(aid)
    assert assigned == all_atom_ids


def test_regions_dont_span_theta_groups():
    """Different theta groups must produce separate regions (geometric
    constraint — rotated and axis-aligned grids cannot share a frame)."""
    case = selected_cases([22])[0][2]  # main θ=0, wing θ=25°
    atoms = atomize(case)
    regions = regionize(case, atoms=atoms)
    atom_by_id = {a.atom_id: a for a in atoms}
    for r in regions:
        thetas = {atom_by_id[aid].theta for aid in r.atom_ids}
        assert len(thetas) == 1, (r.region_id, thetas)


def test_axis_aligned_l_shape_respects_inner_corner_guard():
    case = selected_cases([9])[0][2]
    atoms = atomize(case)
    regions = regionize(case, atoms=atoms)

    for r in regions:
        minx, miny, maxx, maxy = _shape_to_polygon(r.shape).bounds
        if maxx <= 5.0 + 1e-6:
            assert not (
                miny < 5.0 < maxy
            ), (r.region_id, (minx, miny, maxx, maxy))


def test_case_13_十_partitions_via_cross_cut_at_reflex():
    """Cross-cut at inner-corner reflex point should be in the cut history."""
    case = selected_cases([13])[0][2]
    regions = regionize(case)
    labels = {label for r in regions for label in r.cut_history}
    assert "cross_cut" in labels or "vertex_aligned" in labels


def test_region_area_sum_matches_atom_total():
    for idx, _name, case in selected_cases([1, 5, 9, 13, 16, 22, 24, 28]):
        atoms = atomize(case)
        regions = regionize(case, atoms=atoms)
        region_area = sum(_shape_to_polygon(r.shape).area for r in regions)
        assert math.isclose(region_area, _total_atom_area(atoms), rel_tol=1e-3), idx


def test_regionize_runs_on_all_33_cases():
    for idx, _name, case in selected_cases():
        regions = regionize(case)
        assert regions, idx


def test_region_atom_ids_match_actual_atom_union_area():
    case = selected_cases([16])[0][2]
    atoms = atomize(case)
    atom_by_id = {a.atom_id: a for a in atoms}
    regions = regionize(case, atoms=atoms)
    for r in regions:
        atom_polys = [_shape_to_polygon(atom_by_id[aid].shape) for aid in r.atom_ids]
        merged_area = unary_union(atom_polys).area
        region_area = _shape_to_polygon(r.shape).area
        assert math.isclose(merged_area, region_area, rel_tol=1e-3), r.region_id


def test_cut_history_uses_valid_labels():
    case = selected_cases([1])[0][2]
    regions = regionize(case)
    valid_labels = {
        "structural_cross",
        "structural_axis",
        "guard_cross",
        "guard_axis",
        "propagated_axis",
        "cross_cut",
        "vertex_aligned",
        "reflex_pair",
        "axis_mid",
    }
    for r in regions:
        for label in r.cut_history:
            assert label in valid_labels, (r.region_id, label)


def test_target_area_smaller_produces_more_regions():
    case = selected_cases([1])[0][2]
    atoms = atomize(case)
    coarse = regionize(case, atoms=atoms, target_area=12.0)
    fine = regionize(case, atoms=atoms, target_area=4.0)
    assert len(fine) > len(coarse)


def test_disjoint_pieces_get_separate_components():
    """When same-theta pieces aren't connected via the atom graph (e.g.
    fully isolated wings or hole-separated regions), each connected
    component is partitioned independently."""
    case = selected_cases([23])[0][2]  # main + mirror wings (rotated, separate thetas)
    atoms = atomize(case)
    regions = regionize(case, atoms=atoms)
    # Two wing thetas (+30° and -30°) should produce separate regions
    thetas_seen = {round(r.theta, 5) for r in regions}
    assert len(thetas_seen) >= 3  # main 0°, wing1 30°, wing2 60° (-30 mod 90°)


def test_case_28_regions_do_not_cross_curved_transition_vertex():
    case = selected_cases([28])[0][2]
    atoms = atomize(case)
    regions = regionize(case, atoms=atoms)

    for r in regions:
        minx, miny, maxx, maxy = _shape_to_polygon(r.shape).bounds
        assert not (
            minx < 4.0 < maxx and miny < 8.0 < maxy
        ), (r.region_id, (minx, miny, maxx, maxy))


def test_case_28_regions_respect_y4_structural_seam_in_left_leg():
    case = selected_cases([28])[0][2]
    atoms = atomize(case)
    regions = regionize(case, atoms=atoms)

    for r in regions:
        minx, miny, maxx, maxy = _shape_to_polygon(r.shape).bounds
        if maxx <= 4.0 + 1e-6:
            assert not (
                miny < 4.0 < maxy
            ), (r.region_id, (minx, miny, maxx, maxy))


def test_case_15_bottom_bar_reuses_horizontal_cut_axis():
    case = selected_cases([15])[0][2]
    atoms = atomize(case)
    regions = regionize(case, atoms=atoms)

    internal_ys = set()
    for r in regions:
        minx, miny, maxx, maxy = _shape_to_polygon(r.shape).bounds
        if miny >= -1e-6 and maxy <= 5.0 + 1e-6:
            for y in (miny, maxy):
                if 0.0 + 1e-6 < y < 5.0 - 1e-6:
                    internal_ys.add(round(y, 2))

    assert len(internal_ys) <= 1, internal_ys


def test_case_20_regions_do_not_cross_rotated_reflex_axis():
    case = selected_cases([20])[0][2]
    atoms = atomize(case)
    regions = regionize(case, atoms=atoms)
    theta = atoms[0].theta
    guide = build_structural_guides(case)[round(theta, 9)]
    reflex_y = next(y for y in guide.guard_ys if round(y, 2) == 2.01)

    for r in regions:
        minx, miny, maxx, maxy = _local_bounds(r.shape, theta)
        assert not (
            miny < reflex_y - 1e-6 and reflex_y + 1e-6 < maxy
        ), (r.region_id, (minx, miny, maxx, maxy))
