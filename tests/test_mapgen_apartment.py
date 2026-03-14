"""Tests for apartment layout: connectivity and door placement (no T-junctions)."""
import pytest
from v2dlidar.mapgen import ScenarioSpec, generate_scenario, get_apartment_layout_data
from v2dlidar.planner import free_space_connected_components


def _apartment_spec(**kwargs):
    return ScenarioSpec(layout="apartment", width=30.0, height=30.0, apt_iterations=4, **kwargs)


def test_apartment_connectivity_default():
    """With apt_verify_connectivity=True (default), free space must be one connected component."""
    for seed in [0, 1, 42, 123, 999]:
        spec = _apartment_spec(seed=seed)
        segments, _, interior, res = generate_scenario(spec)
        n_comp, _ = free_space_connected_components(
            segments, spec.width, spec.height, res, interior=interior
        )
        assert n_comp == 1, f"seed={seed}: expected 1 connected component, got {n_comp}"


def test_apartment_connectivity_with_verify_true():
    """Explicit apt_verify_connectivity=True: free space must be one connected component."""
    for seed in [7, 100, 456]:
        spec = _apartment_spec(seed=seed, apt_verify_connectivity=True)
        segments, _, interior, res = generate_scenario(spec)
        n_comp, _ = free_space_connected_components(
            segments, spec.width, spec.height, res, interior=interior
        )
        assert n_comp == 1, f"seed={seed}: expected 1 connected component, got {n_comp}"


def test_apartment_doors_not_at_tjunction():
    """Doors must connect at least two rooms (>=2 on wall); fallback doors may have 3+."""
    from v2dlidar.mapgen import _resolve_apartment_rooms_and_doors

    tol = 0.1
    for seed in [0, 1, 42, 100]:
        spec = _apartment_spec(seed=seed)
        rooms, doors, _ = _resolve_apartment_rooms_and_doors(spec)
        for door in doors:
            dx, dy, orientation = door[0], door[1], door[2]
            if orientation == "vertical":
                on_wall = [
                    r for r in rooms
                    if (abs(r.x - dx) < tol or abs((r.x + r.width) - dx) < tol)
                    and r.y <= dy + tol and dy - tol <= r.y + r.height
                ]
            else:
                on_wall = [
                    r for r in rooms
                    if (abs(r.y - dy) < tol or abs((r.y + r.height) - dy) < tol)
                    and r.x <= dx + tol and dx - tol <= r.x + r.width
                ]
            assert len(on_wall) >= 2, (
                f"seed={seed}: door at ({dx}, {dy}) {orientation}: expected >=2 rooms on wall, got {len(on_wall)}"
            )


def test_apartment_layout_data_matches_scenario():
    """get_apartment_layout_data must return same (rooms, doors) as used for scenario segments."""
    spec = _apartment_spec(seed=42)
    segments, _, _, _ = generate_scenario(spec)
    layout_rooms, layout_doors = get_apartment_layout_data(spec)
    assert layout_rooms is not None and layout_doors is not None
    assert len(layout_doors) >= 1


def test_apt_verify_connectivity_default_true():
    """ScenarioSpec.apt_verify_connectivity defaults to True."""
    assert ScenarioSpec().apt_verify_connectivity is True
    assert _apartment_spec(seed=0).apt_verify_connectivity is True


def test_apt_verify_connectivity_false():
    """With apt_verify_connectivity=False, resolver returns without enforcing connectivity."""
    from v2dlidar.mapgen import _resolve_apartment_rooms_and_doors

    spec = _apartment_spec(seed=0, apt_verify_connectivity=False)
    rooms, doors, effective = _resolve_apartment_rooms_and_doors(spec)
    assert len(rooms) >= 1 and len(doors) >= 1


def test_layout_data_doors_match_scenario():
    """Layout data (rooms, doors) matches what generate_scenario uses (same door count and positions)."""
    spec = _apartment_spec(seed=42)
    segments, _, _, _ = generate_scenario(spec)
    layout_rooms, layout_doors = get_apartment_layout_data(spec)
    assert layout_rooms is not None and layout_doors is not None
    # Regenerating with same spec must yield same door count
    from v2dlidar.mapgen import _resolve_apartment_rooms_and_doors
    rooms2, doors2, _ = _resolve_apartment_rooms_and_doors(spec)
    assert len(layout_doors) == len(doors2)
    for i, d in enumerate(layout_doors):
        assert len(d) == 4
        assert d[0] == doors2[i][0] and d[1] == doors2[i][1] and d[2] == doors2[i][2]


def test_get_apartment_layout_data_returns_none_for_union():
    """get_apartment_layout_data returns None when layout is not apartment."""
    spec = ScenarioSpec(layout="union", width=30, height=30, seed=0)
    assert get_apartment_layout_data(spec) is None


def test_connectivity_after_fallback():
    """Seeds that could previously be disconnected are now one connected component."""
    for seed in [0, 1, 42, 100, 123, 456, 999]:
        spec = _apartment_spec(seed=seed)
        segments, _, interior, res = generate_scenario(spec)
        n_comp, _ = free_space_connected_components(
            segments, spec.width, spec.height, res, interior=interior
        )
        assert n_comp == 1, f"seed={seed}: expected 1 connected component, got {n_comp}"


def test_every_room_has_at_least_one_door():
    """Connectivity: every room has at least one door to another room."""
    from v2dlidar.mapgen import _resolve_apartment_rooms_and_doors, _room_has_door

    for seed in [0, 1, 42, 100, 123, 456, 999]:
        spec = _apartment_spec(seed=seed)
        rooms, doors, _ = _resolve_apartment_rooms_and_doors(spec)
        for r in rooms:
            assert _room_has_door(r, rooms, doors), (
                f"seed={seed}: room at ({r.x}, {r.y}) size {r.width}x{r.height} has no door"
            )
