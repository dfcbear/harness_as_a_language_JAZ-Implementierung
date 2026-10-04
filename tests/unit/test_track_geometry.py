"""
Unit tests for track_geometry module.
Run with:  pytest tests/unit/test_track_geometry.py -v
"""

import math
import pytest
import random
import sys
import os

# Ensure the racing directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from racing.track_geometry import (
    Vec2, Checkpoint, Track, TrackIndex, NearestResult,
    vadd, vsub, vscale, vdot, vcross, vlen, vnorm, lerp,
    build_track, build_index, make_oval, make_figure8,
    nearest_point, signed_lateral_offset, is_off_track,
    crossed_gate, update_lap_progress, start_pose,
    progress_at, point_ahead, curvature_ahead,
    _catmull_rom_point, _catmull_rom_tangent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def oval():
    return make_oval(rx=300, ry=200, n_points=12, half_width=10,
                     samples_per_segment=20, num_checkpoints=4)


@pytest.fixture
def fig8():
    return make_figure8(r=250, n_points=24, half_width=7, num_checkpoints=6)


# ---------------------------------------------------------------------------
# Vec2 helpers
# ---------------------------------------------------------------------------

class TestVec2:
    def test_add(self):
        assert vadd((1, 2), (3, 4)) == (4, 6)

    def test_sub(self):
        assert vsub((5, 7), (2, 3)) == (3, 4)

    def test_scale(self):
        assert vscale((2, 3), 2) == (4, 6)

    def test_dot(self):
        assert vdot((1, 0), (0, 1)) == 0
        assert vdot((1, 2), (3, 4)) == 11

    def test_cross(self):
        assert vcross((1, 0), (0, 1)) == 1
        assert vcross((0, 1), (1, 0)) == -1

    def test_len(self):
        assert vlen((3, 4)) == 5.0

    def test_norm(self):
        n = vnorm((3, 4))
        assert abs(vlen(n) - 1.0) < 1e-9
        assert vnorm((0, 0)) == (0, 0)

    def test_lerp(self):
        assert lerp((0, 0), (10, 20), 0.5) == (5, 10)


# ---------------------------------------------------------------------------
# Track building
# ---------------------------------------------------------------------------

class TestBuildTrack:
    def test_centerline_length(self, oval):
        track, idx = oval
        assert len(track.centerline) == 12 * 20

    def test_total_length_positive(self, oval):
        track, idx = oval
        assert idx.total > 0

    def test_checkpoints_count(self, oval):
        track, idx = oval
        # 4 intermediate + 1 finish = 5
        assert len(track.checkpoints) == 5

    def test_start_line_exists(self, oval):
        track, idx = oval
        assert track.start_line is not None
        assert len(track.start_line) == 2

    def test_too_few_control_points(self):
        with pytest.raises(ValueError):
            build_track([(0, 0), (1, 1)])

    def test_closed_loop(self, oval):
        track, idx = oval
        m = len(track.centerline)
        d = vlen(vsub(track.centerline[0], track.centerline[-1]))
        # The last sample should be close to the first (closed Catmull-Rom)
        assert d < vlen(vsub(track.centerline[0], track.centerline[1])) * 2

    def test_normals_are_unit(self, oval):
        track, idx = oval
        for n in idx.normals:
            assert abs(vlen(n) - 1.0) < 1e-6

    def test_tangents_are_unit(self, oval):
        track, idx = oval
        for t in idx.tangents:
            assert abs(vlen(t) - 1.0) < 1e-6

    def test_normal_is_left_of_tangent(self, oval):
        track, idx = oval
        for i in range(len(idx.tangents)):
            t = idx.tangents[i]
            n = idx.normals[i]
            # left normal = (-t.y, t.x)
            assert abs(n[0] - (-t[1])) < 1e-6
            assert abs(n[1] - t[0]) < 1e-6


# ---------------------------------------------------------------------------
# Nearest point
# ---------------------------------------------------------------------------

class TestNearestPoint:
    def test_point_on_centerline(self, oval):
        track, idx = oval
        p = track.centerline[50]
        nr = nearest_point(track, idx, p)
        assert nr.distance < 0.5

    def test_point_off_centerline(self, oval):
        track, idx = oval
        n = idx.normals[50]
        p = vadd(track.centerline[50], vscale(n, 5.0))
        nr = nearest_point(track, idx, p)
        assert 4.0 < nr.distance < 6.0

    def test_grid_matches_brute_force(self, oval):
        track, idx = oval
        random.seed(42)
        for _ in range(100):
            q = (random.uniform(-400, 400), random.uniform(-300, 300))
            nr_grid = nearest_point(track, idx, q, use_grid=True)
            nr_brute = nearest_point(track, idx, q, use_grid=False)
            assert abs(nr_grid.distance - nr_brute.distance) < 1.0

    def test_nearest_result_fields(self, oval):
        track, idx = oval
        nr = nearest_point(track, idx, track.centerline[10])
        assert 0 <= nr.seg_index < len(track.centerline)
        assert 0.0 <= nr.t <= 1.0
        assert nr.distance >= 0.0
        assert nr.side in (-1.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Off-track detection
# ---------------------------------------------------------------------------

class TestOffTrack:
    def test_on_centerline_is_on_track(self, oval):
        track, idx = oval
        assert not is_off_track(track, idx, track.centerline[100])

    def test_far_off_is_off_track(self, oval):
        track, idx = oval
        n = idx.normals[100]
        off = vadd(track.centerline[100], vscale(n, 50))
        assert is_off_track(track, idx, off)

    def test_within_half_width_is_on_track(self, oval):
        track, idx = oval
        n = idx.normals[100]
        within = vadd(track.centerline[100], vscale(n, 8))
        assert not is_off_track(track, idx, within)

    def test_just_outside_half_width(self, oval):
        track, idx = oval
        n = idx.normals[100]
        outside = vadd(track.centerline[100], vscale(n, 11))
        assert is_off_track(track, idx, outside)

    def test_car_half_width_margin(self, oval):
        track, idx = oval
        n = idx.normals[100]
        # 10 half_width + 2 car_half_width = 12, point at 11 should be on track
        pt = vadd(track.centerline[100], vscale(n, 11))
        assert not is_off_track(track, idx, pt, car_half_width=2.0)
        pt2 = vadd(track.centerline[100], vscale(n, 13))
        assert is_off_track(track, idx, pt2, car_half_width=2.0)


# ---------------------------------------------------------------------------
# Signed lateral offset
# ---------------------------------------------------------------------------

class TestSignedLateralOffset:
    def test_left_is_positive(self, oval):
        track, idx = oval
        n = idx.normals[50]
        left = vadd(track.centerline[50], vscale(n, 5))
        assert signed_lateral_offset(track, idx, left) > 0

    def test_right_is_negative(self, oval):
        track, idx = oval
        n = idx.normals[50]
        right = vsub(track.centerline[50], vscale(n, 5))
        assert signed_lateral_offset(track, idx, right) < 0

    def test_centerline_is_zero(self, oval):
        track, idx = oval
        assert abs(signed_lateral_offset(track, idx, track.centerline[50])) < 1.0


# ---------------------------------------------------------------------------
# Gate crossing
# ---------------------------------------------------------------------------

class TestCrossedGate:
    def test_crosses_gate(self, oval):
        track, idx = oval
        gate = track.checkpoints[0]
        t0 = idx.tangents[0]
        behind = vsub(track.centerline[0], vscale(t0, 5))
        ahead = vadd(track.centerline[0], vscale(t0, 5))
        assert crossed_gate(behind, ahead, gate.a, gate.b)

    def test_does_not_cross_parallel(self, oval):
        track, idx = oval
        gate = track.checkpoints[0]
        t0 = idx.tangents[0]
        n0 = idx.normals[0]
        beside1 = vadd(vsub(track.centerline[0], vscale(t0, 5)), vscale(n0, 30))
        beside2 = vadd(vadd(track.centerline[0], vscale(t0, 5)), vscale(n0, 30))
        assert not crossed_gate(beside1, beside2, gate.a, gate.b)

    def test_no_cross_when_both_on_same_side(self, oval):
        track, idx = oval
        gate = track.checkpoints[0]
        t0 = idx.tangents[0]
        both_behind_a = vsub(track.centerline[0], vscale(t0, 10))
        both_behind_b = vsub(track.centerline[0], vscale(t0, 3))
        assert not crossed_gate(both_behind_a, both_behind_b, gate.a, gate.b)


# ---------------------------------------------------------------------------
# Lap progress state machine
# ---------------------------------------------------------------------------

class TestLapProgress:
    def test_initial_state_no_lap(self, oval):
        track, idx = oval
        # Just driving around without crossing gates
        ci, ef, done = update_lap_progress(
            track, (0, 0), (1, 0), checkpoint_index=1, expecting_finish=False
        )
        assert not done

    def test_full_lap(self, oval):
        track, idx = oval
        ci = 1
        ef = False
        laps = 0

        # Cross all intermediate checkpoints in order
        for i in range(1, len(track.checkpoints)):
            gate = track.checkpoints[i]
            tangent = idx.tangents[gate.index]
            prev = vsub(gate.center, vscale(tangent, 5))
            cur = vadd(gate.center, vscale(tangent, 5))
            ci, ef, done = update_lap_progress(track, prev, cur, ci, ef)
            assert not done, f"Lap completed early at checkpoint {i}"

        # Cross finish line
        gate0 = track.checkpoints[0]
        tangent0 = idx.tangents[0]
        prev0 = vsub(gate0.center, vscale(tangent0, 5))
        cur0 = vadd(gate0.center, vscale(tangent0, 5))
        ci, ef, done = update_lap_progress(track, prev0, cur0, ci, ef)
        assert done
        assert ci == 1
        assert not ef
        laps += 1
        assert laps == 1

    def test_no_lap_without_checkpoints(self, oval):
        track, idx = oval
        # Cross finish line without passing checkpoints
        gate0 = track.checkpoints[0]
        tangent0 = idx.tangents[0]
        prev0 = vsub(gate0.center, vscale(tangent0, 5))
        cur0 = vadd(gate0.center, vscale(tangent0, 5))
        ci, ef, done = update_lap_progress(track, prev0, cur0, 1, False)
        # checkpoint_index=1, not 0, so finish line is not the expected gate
        assert not done

    def test_two_laps(self, oval):
        track, idx = oval
        ci = 1
        ef = False
        total_laps = 0

        for lap in range(2):
            for i in range(1, len(track.checkpoints)):
                gate = track.checkpoints[i]
                tangent = idx.tangents[gate.index]
                prev = vsub(gate.center, vscale(tangent, 5))
                cur = vadd(gate.center, vscale(tangent, 5))
                ci, ef, done = update_lap_progress(track, prev, cur, ci, ef)
                assert not done
            gate0 = track.checkpoints[0]
            tangent0 = idx.tangents[0]
            prev0 = vsub(gate0.center, vscale(tangent0, 5))
            cur0 = vadd(gate0.center, vscale(tangent0, 5))
            ci, ef, done = update_lap_progress(track, prev0, cur0, ci, ef)
            assert done
            total_laps += 1

        assert total_laps == 2


# ---------------------------------------------------------------------------
# Start pose
# ---------------------------------------------------------------------------

class TestStartPose:
    def test_returns_position_and_heading(self, oval):
        track, idx = oval
        pos, heading = start_pose(track, idx)
        assert len(pos) == 2
        assert isinstance(heading, float)

    def test_lateral_offset(self, oval):
        track, idx = oval
        pos, heading = start_pose(track, idx, lateral_offset=5.0)
        n = idx.normals[0]
        expected = vadd(track.centerline[0], vscale(n, 5.0))
        assert abs(pos[0] - expected[0]) < 1.0
        assert abs(pos[1] - expected[1]) < 1.0

    def test_backward_offset(self, oval):
        track, idx = oval
        pos, heading = start_pose(track, idx, backward=15.0)
        t = idx.tangents[0]
        expected = vsub(track.centerline[0], vscale(t, 15.0))
        assert abs(pos[0] - expected[0]) < 1.0
        assert abs(pos[1] - expected[1]) < 1.0

    def test_heading_matches_tangent(self, oval):
        track, idx = oval
        _, heading = start_pose(track, idx)
        t0 = idx.tangents[0]
        expected = math.atan2(t0[1], t0[0])
        assert abs(heading - expected) < 1e-6


# ---------------------------------------------------------------------------
# Progress and look-ahead
# ---------------------------------------------------------------------------

class TestProgressAndLookahead:
    def test_progress_at_start(self, oval):
        track, idx = oval
        p = track.centerline[0]
        prog = progress_at(track, idx, p)
        assert 0.0 <= prog < 1.0

    def test_progress_increases(self, oval):
        track, idx = oval
        p1 = track.centerline[0]
        p2 = track.centerline[10]
        prog1 = progress_at(track, idx, p1)
        prog2 = progress_at(track, idx, p2)
        assert prog2 > prog1 or prog2 < 0.1  # wrap-around

    def test_point_ahead(self, oval):
        track, idx = oval
        p = track.centerline[0]
        ahead = point_ahead(track, idx, p, distance=50.0)
        assert vlen(vsub(ahead, p)) > 10

    def test_curvature_finite(self, oval):
        track, idx = oval
        kappa = curvature_ahead(track, idx, track.centerline[0], distance=20.0)
        assert math.isfinite(kappa)
        assert kappa >= 0


# ---------------------------------------------------------------------------
# Figure-8 track
# ---------------------------------------------------------------------------

class TestFigure8:
    def test_builds(self, fig8):
        track, idx = fig8
        assert len(track.centerline) == 24 * 20
        assert idx.total > 0

    def test_checkpoints(self, fig8):
        track, idx = fig8
        assert len(track.checkpoints) == 7  # 6 + finish

    def test_nearest_point(self, fig8):
        track, idx = fig8
        p = track.centerline[100]
        nr = nearest_point(track, idx, p)
        assert nr.distance < 0.5

    def test_off_track(self, fig8):
        track, idx = fig8
        n = idx.normals[100]
        off = vadd(track.centerline[100], vscale(n, 50))
        assert is_off_track(track, idx, off)


# ---------------------------------------------------------------------------
# Catmull-Rom
# ---------------------------------------------------------------------------

class TestCatmullRom:
    def test_endpoint_at_t0(self):
        p0, p1, p2, p3 = (0, 0), (1, 0), (2, 0), (3, 0)
        result = _catmull_rom_point(p0, p1, p2, p3, 0.0)
        assert abs(result[0] - 1.0) < 1e-6
        assert abs(result[1] - 0.0) < 1e-6

    def test_tangent_at_t0(self):
        p0, p1, p2, p3 = (0, 0), (1, 0), (2, 0), (3, 0)
        result = _catmull_rom_tangent(p0, p1, p2, p3, 0.0)
        # Tangent should point in +x direction
        assert result[0] > 0

    def test_symmetry(self):
        # Symmetric control points should produce symmetric curve
        p0, p1, p2, p3 = (-2, 0), (-1, 1), (1, 1), (2, 0)
        pt1 = _catmull_rom_point(p0, p1, p2, p3, 0.5)
        # At t=0.5, should be at the midpoint
        assert abs(pt1[0]) < 1e-6
