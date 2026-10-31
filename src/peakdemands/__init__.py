"""peakdemands — bursts, most demanding scenarios and GPS 3.0 exposure metrics
from tracking data (GPS, LPS, optical).

    import peakdemands as pk
    kin = pk.from_positions(time, x, y)            # or pk.from_speed(time, speed)
    sprints = pk.sprints(kin, 7.0)
    curve = pk.peak_curve(kin, "distance")         # rolling MDS, fixed epochs, power law
    from peakdemands.report import session_report
"""

from .datasets import load_metrica, simulate_session
from .efforts import (Effort, Sequence, accelerations, decelerations, detect, near_mss_exposures,
                      repeated_sequences, repeated_sprint_bouts, sprints, turns)
from .gps3 import (bundle_mean_speed, bundle_speed, directional_share, estimated_max_mp1min,
                   estimated_max_s1min, intensity_exposure_time, velocity_change_power, weekly_iet)
from .kinematics import Kinematics, from_positions, from_speed
from .mds import DURATIONS, PeakCurve, peak_curve, rolling_mean

__version__ = "0.1.0"
