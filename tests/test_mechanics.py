"""Direction-sensitive work: exact answers on simple trajectories (per kg, v = 5 m/s)."""

import numpy as np
import pytest

import peakdemands as pk

FS = 25.0


def path(speed, heading):
    t = np.arange(speed.size) / FS
    return pk.from_positions(t, np.cumsum(speed * np.cos(heading)) / FS, np.cumsum(speed * np.sin(heading)) / FS)


T = np.arange(0, 14, 1 / FS)
V = np.full(T.size, 5.0)


def test_straight_constant_speed_no_work():
    m = pk.mechanical_work(path(V, np.zeros(T.size)))
    assert m.w_total == pytest.approx(0, abs=0.5) and not m.manoeuvres


def test_quarter_turn_constant_speed():
    m = pk.mechanical_work(path(V, np.clip((T - 5), 0, 1) * np.pi / 2))
    assert m.w_direction == pytest.approx(25, rel=0.08)      # v²: ½v² absorbed + ½v² produced
    assert m.w_speed < 2
    assert [(x.family, x.side) for x in m.manoeuvres] == [("turn", "left")]


def test_right_turn_is_mirror():
    left = pk.mechanical_work(path(V, np.clip((T - 5), 0, 1) * np.pi / 2))
    right = pk.mechanical_work(path(V, -np.clip((T - 5), 0, 1) * np.pi / 2))
    assert right.w_direction == pytest.approx(left.w_direction, rel=1e-6)
    assert right.manoeuvres[0].side == "right"


def test_u_turn_on_a_curve():
    m = pk.mechanical_work(path(V, np.clip((T - 5) / 3, 0, 1) * np.pi))
    assert m.w_direction == pytest.approx(50, rel=0.05)      # 2v²


def test_stop_and_go_cut_is_speed_work():
    sp = np.interp(T, [0, 4, 5, 6, 14], [5, 5, 0, 5, 5])
    m = pk.mechanical_work(path(sp, np.where(T < 5, 0, np.pi)))
    assert m.w_speed == pytest.approx(25, rel=0.05)          # ½v² braking + ½v² re-accelerating
    assert m.w_direction == pytest.approx(0, abs=1)
    assert [x.family for x in m.manoeuvres] == ["deceleration", "acceleration"]


def test_straight_acceleration():
    sp = np.interp(T, [0, 2, 6, 14], [0, 0, 8, 8])
    m = pk.mechanical_work(path(sp, np.zeros(T.size)))
    assert m.w_speed == pytest.approx(32, rel=0.03)          # ½·8²
    assert m.w_direction == 0


@pytest.mark.parametrize("fs,noise", [(25, 0.05), (10, 0.05)])
def test_deadband_controls_noise(fs, noise):
    rng = np.random.default_rng(1)
    t = np.arange(0, 600, 1 / fs)
    v = 2 + 1.5 * np.sin(2 * np.pi * t / 20) + np.where((t % 60) < 4, 4 * np.sin(np.pi * (t % 60) / 4), 0)
    true = np.abs(np.diff(0.5 * v ** 2)).sum()
    x = np.cumsum(v) / fs + rng.normal(0, noise, t.size)
    k = pk.from_positions(t, x, rng.normal(0, noise, t.size))
    raw = pk.mechanical_work(k, deadband=0).w_speed
    filt = pk.mechanical_work(k).w_speed
    assert raw > 1.5 * true                                   # noise inflates the naive sum
    assert filt == pytest.approx(true, rel=0.08)


def test_speed_only_has_no_direction_term():
    k = pk.from_speed(T, np.interp(T, [0, 2, 6, 14], [0, 0, 8, 8]))
    m = pk.mechanical_work(k)
    assert m.w_direction == 0 and m.w_speed == pytest.approx(32, rel=0.03)


def test_mechanical_bursts_and_mds():
    m = pk.mechanical_work(path(V, np.clip((T - 5), 0, 1) * np.pi / 2))
    k = path(V, np.clip((T - 5), 0, 1) * np.pi / 2)
    b = pk.mechanical_bursts(k, m.power, 10.0)
    assert len(b) == 1 and 4.5 < b[0].start < 6
    c = pk.peak_curve(k, "power", durations=(1, 2))
    assert c.peak[0] > c.peak[1] > 0
