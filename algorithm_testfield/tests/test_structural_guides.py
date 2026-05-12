from celllayout_tf.cases import selected_cases
from celllayout_tf.structural_guides import build_structural_guides


def test_case_28_curved_transition_vertices_are_guides():
    case = selected_cases([28])[0][2]
    guides = build_structural_guides(case)
    guide = guides[0.0]

    assert (4.0, 8.0) in guide.points
    assert (8.0, 4.0) in guide.points
    assert (4.0, 8.0) in guide.event_points
    assert (8.0, 4.0) in guide.event_points
    assert 4.0 in guide.xs
    assert 8.0 in guide.ys
    assert 4.0 in guide.event_xs
    assert 8.0 in guide.event_ys


def test_case_20_rotated_reflex_vertex_is_guard_guide():
    case = selected_cases([20])[0][2]
    guides = build_structural_guides(case)
    guide = next(iter(guides.values()))

    assert round(guide.theta, 6) == round(next(iter(guides)), 6)
    assert any(round(y, 2) == 2.01 for y in guide.guard_ys)
    assert any(
        round(x, 2) == 6.72 and round(y, 2) == 2.01
        for x, y in guide.guard_points
    )


def test_circle_guides_stay_sparse():
    case = selected_cases([25])[0][2]
    guides = build_structural_guides(case)
    guide = guides[0.0]

    assert len(guide.points) <= 8
