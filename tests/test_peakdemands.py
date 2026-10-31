import numpy as np
import pytest

import peakdemands as pk


@pytest.fixture(scope="module")
def sim():
    s = pk.simulate_session()
    return s, pk.from_positions(s["time"], s["x"], s["y"])


def test_speed_recovered(sim):
    s, k = sim
    err = np.nanmedian(np.abs(k.speed - s["true_speed"]))
    assert err < 0.15


def test_constant_turn_is_centripetal():
    fs, v, w = 10.0, 6.0, 0.5               # 6 m/s on a circle at 0.5 rad/s: a_c = 3 m/s², a_t = 0
    t = np.arange(0, 60, 1 / fs)
    x, y = v / w * np.sin(w * t), v / w * (1 - np.cos(w * t))
    k = pk.from_positions(t, x, y)
    mid = slice(50, -50)
    assert np.nanmedian(k.a_centripetal[mid]) == pytest.approx(3.0, rel=0.02)
    assert np.nanmedian(np.abs(k.a_tangential[mid])) < 0.05
    assert pk.directional_share(k) > 0.95


def test_straight_acceleration_is_tangential():
    fs = 10.0
    t = np.arange(0, 10, 1 / fs)
    x = 0.5 * 2.0 * t ** 2                  # constant 2 m/s² on a line
    k = pk.from_speed(t, 2.0 * t)
    k2 = pk.from_positions(t, x, np.zeros_like(x))
    assert np.nanmedian(k.a_tangential[10:-10]) == pytest.approx(2.0, rel=0.01)
    assert np.nanmedian(k2.a_tangential[10:-10]) == pytest.approx(2.0, rel=0.01)
    assert np.nanmax(k2.a_centripetal[10:-10]) < 1e-6


def test_sprints_found(sim):
    s, k = sim
    sp = pk.sprints(k, 7.0)
    starts = [e.start for e in sp]
    for true in s["sprint_starts"]:
        assert any(abs(st - true) < 2.5 for st in starts)


def test_repeated_sequences(sim):
    s, k = sim
    seq = pk.repeated_sequences(pk.sprints(k, 7.0), max_recovery=60)
    assert any(q.count >= 3 and abs(q.start - 100) < 3 for q in seq)   # sprints at 100, 130, 160 s
    assert pk.repeated_sprint_bouts(pk.sprints(k, 7.0)) == []           # 26 s recoveries: not Spencer bouts


def test_spencer_bout():
    E = pk.Effort
    ev = [E("sprint", s, s + 3, 8, 5, 8, 20) for s in (0, 20, 40, 300)]
    b = pk.repeated_sprint_bouts(ev)
    assert len(b) == 1 and b[0].count == 3 and b[0].mean_recovery == pytest.approx(17)


def test_rolling_beats_fixed_and_power_law():
    fs = 10.0
    t = np.arange(0, 1800, 1 / fs)
    v = np.full(t.size, 1.5)
    v[(t >= 125) & (t < 185)] = 5.0          # a hard minute straddling two fixed epochs
    k = pk.from_speed(t, v)
    c = pk.peak_curve(k, "distance", durations=(30, 60, 120, 300))
    i = list(c.durations).index(60)
    assert c.peak[i] == pytest.approx(300, rel=0.03)      # 5 m/s = 300 m/min
    assert c.fixed[i] < c.peak[i]
    assert c.peak_start[i] == pytest.approx(125, abs=1.5)
    cc, n, r2 = c.power_law()
    assert n < 0


def test_bundle_equations():
    assert pk.bundle_mean_speed(60, 0, 1) == pytest.approx(0.695, abs=0.002)   # Equation 1 constant
    assert pk.estimated_max_s1min(16, 32) == pytest.approx(16 + 0.695 * 16)
    assert pk.estimated_max_mp1min(16, 32) == pytest.approx(1.045 * (16 / 3.6 + 0.695 * 16 / 3.6))
    assert pk.bundle_speed(0, 16, 32) == 32


def test_iet():
    fs = 10.0
    t = np.arange(0, 600, 1 / fs)
    v = np.where((t >= 100) & (t < 280), 6.0, 2.0)
    k = pk.from_speed(t, v)
    iet = pk.intensity_exposure_time(k, np.nan_to_num(k.speed), 6.0, thresholds=(0.8,))
    # rolling mean 2 + 4·f ≥ 4.8 m/s when the window holds ≥ 70 % of the hard part: from 142 to 298 s
    assert iet[0.8] == pytest.approx(156 / 60, abs=0.05)
    assert pk.weekly_iet([iet, iet])[0.8] == pytest.approx(2 * iet[0.8])


def test_report(sim, tmp_path):
    pytest.importorskip("weasyprint")
    from peakdemands.report import session_report
    s, k = sim
    for lang in ("fr", "en"):
        out = session_report(k, tmp_path / f"r_{lang}.pdf", mss=8.5, mas=16.0, lang=lang)
        assert out.read_bytes()[:4] == b"%PDF"
