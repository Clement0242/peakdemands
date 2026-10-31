# peakdemands

Bursts, most demanding scenarios (MDS) and GPS 3.0 exposure metrics from
tracking data — GPS, LPS or optical — with a designed PDF report.

```python
import peakdemands as pk

kin = pk.from_positions(time, x, y)        # x, y in metres; or pk.from_speed(time, speed)
sprints = pk.sprints(kin, 7.0)             # efforts > 7 m/s held ≥ 1 s
curve = pk.peak_curve(kin, "distance")     # rolling peaks for 15 s … 10 min, fixed epochs, power law
from peakdemands.report import session_report
session_report(kin, "match.pdf", athlete="Player 9", mss=9.2, mas=17.0)
```

**Full match report for physical preparation: [EN](docs/example_match_report_en.pdf) ·
[FR](docs/example_match_report_fr.pdf)** — a whole team over a real professional match (open optical tracking +
event data). **Single-player session report: [EN](docs/example_match_en.pdf) · [FR](docs/example_match_fr.pdf).**

```python
from peakdemands.match import load_metrica_match
from peakdemands.report.match import match_report
match_report(load_metrica_match(game=2, team="Home"), "match.pdf")
```

The match report crosses everything: team table (minutes, distance, m/min, HSR, sprints, peak speed,
accelerations, repeated sequences), average positions and sprint map, load **by game context** (in
possession, out of possession, ball dead, from the event data), 5-minute heatmap and half-to-half
change, team intensity with goals, peak demands per player and **references for drills** (median and
90th percentile of the 1-, 3- and 5-min peaks), direction-sensitive mechanical load with the team
movement signature and left/right balance, and **leads for the week** from transparent rules
(recovery priority, end-of-match drop, top-up targets for substitutes, speed exposure, asymmetry).

## What it does, and where each definition comes from

| | Definition | Source |
|---|---|---|
| **Complete acceleration** | From x, y: tangential (speed change) and centripetal (turning) components, \|a\|² = a<sub>t</sub>² + a<sub>c</sub>². Savitzky-Golay smoothing, tracking jumps removed. | Buchheit & Simpson 2017; Buchheit et al. 2026 (GPS 3.0) |
| **Mechanical work and power** | Speed-change + direction-change work of the centre of mass, manoeuvres and movement signature (below). | Gray et al. 2018; Cavagna et al. 1976 |
| **Efforts (bursts)** | Runs above a threshold held ≥ 1 s (speed) or ≥ 0.5 s (acceleration); sprints, high-speed efforts, accelerations, decelerations, turns. | Harper et al. 2019 (> 2.5 m/s²) |
| **Near-to-MSS exposures** | Efforts above 85, 90, 95 % of maximal sprinting speed. | Buchheit et al. 2023 |
| **Repeated-sprint sequences** | ≥ 2 efforts no more than 60 s apart. | Buchheit et al. 2010 |
| **Repeated-sprint bouts** | ≥ 3 sprints, mean recovery < 21 s. | Spencer et al. 2004 |
| **Most demanding scenarios** | Peak of the rolling average per window length (15 s → 10 min), with the fixed-epoch value, the time of the peak, the next window and the number of separate passages ≥ 90 % of the peak. Distance, high-speed distance, acc + dec per minute. | Varley et al. 2012; Fereday et al. 2020; Whitehead et al. 2018; Novak et al. 2021 |
| **Intensity-duration law** | I = c·t<sup>n</sup>, least squares in log-log space. | Delaney et al. 2018 |
| **Capacity anchors** | Bundle speed-duration model; Estimated Maximal S1min = MAS + 0.695·(MSS − MAS) (Eq. 1); Estimated Maximal MP1min = 1.045·[MAS/3.6 + 0.695·(MSS − MAS)/3.6] (Eq. 2). | Bundle et al. 2003; Buchheit et al. 2026 |
| **IET / WIET** | Minutes with the 60-s rolling average above 60/70/80 % of a reference; summed over a week. | Buchheit et al. 2026 |

## Direction-sensitive mechanical load

GPS 3.0 asks for mechanical work and power that count changes of **direction** as well as
changes of speed. The analyser used in the paper (ADI, Hudl) is proprietary and its algorithm
is not published, so `peakdemands.mechanics` builds an **open model from published physics**:

* **Speed-change work** — the change in kinetic energy of the centre of mass between samples,
  summed in absolute value (positive when accelerating, negative when braking): the
  horizontal work of Gray et al. (2018). Blind to turning by construction.
* **Direction-change work** — kinetic energy counted separately along and across the athlete's
  heading at the start of each manoeuvre (the per-axis accounting of Cavagna et al. 1976). A
  turn moves kinetic energy from one axis to the other: it is absorbed along the old direction
  and produced along the new one.
* **Noise** — summing |ΔKE| also sums tracking noise (×2 on a simulated run with 5 cm noise);
  a 0.2 m/s hysteresis filter brings the error back to about 5 %.

| Movement (v = 5 m/s) | Speed work | Direction work | Test |
|---|---|---|---|
| Straight, constant speed | 0 | 0 | ✓ |
| Straight acceleration 0 → 8 m/s | ½·8² = 32 J/kg | 0 | ✓ |
| Stop-and-go 180° cut | v² = 25 J/kg | 0 | ✓ |
| 90° turn at constant speed | ≈ 0 | v² = 25 J/kg | ✓ |
| U-turn on a curve at constant speed | ≈ 0 | 2v² = 50 J/kg | ✓ |

On top of it: **mechanical power** (W/kg) and its most demanding scenarios, **mechanical
bursts** (accelerations, brakes and cuts alike), **intensity exposure time** on power, and a
**movement signature**: every manoeuvre classified as acceleration, deceleration, turn,
curved acceleration or curved braking, left or right, with its work.

On the open match, the direction part is **24 % of the horizontal work (30 % within
manoeuvres)**, stable whatever the smoothing — the order of magnitude Gray reports for
direction change in meaningful manoeuvres (≈ 35 %). These are this model's numbers: they are
not ADI watts and are not comparable with them or with the paper's Estimated Maximal MP1min;
the model is horizontal and centre-of-mass only (no vertical, limb or air work).

## Example: a real match

`load_metrica()` downloads the Metrica Sports sample data (an anonymised professional
match, optical tracking at 25 Hz) on first use; the files are not redistributed here.
For one midfielder: 9.7 km, 7 sprints above 7 m/s, peak 1-min distance 186 m/min (1.8 × the
match average), fixed epochs underestimating the peaks by 0.3–5 %, 859 manoeuvres and
**24 % of the horizontal mechanical work coming from direction changes** — the part classic
GPS metrics leave out.

![MDS curves and complete acceleration](docs/mds_direction.png)

## Install

```bash
pip install "peakdemands[report] @ git+https://github.com/Clement0242/peakdemands"
```

PDF reports use WeasyPrint (Pango needed: `apt install libpango-1.0-0 libpangoft2-1.0-0`).
Typeface: Jost (SIL Open Font License, shipped).

## Tests

20 tests. Ground truths on synthetic signals: the five movements of the table above, the
noise filter, a constant-speed circle (pure centripetal acceleration,
3 m/s² recovered within 2 %), a straight constant acceleration, a simulated session with
known sprints, a hard minute straddling two fixed epochs (the rolling peak finds it, fixed
epochs do not), Bundle's 0.695 constant, IET on a known profile, Spencer bouts. Run
`pytest`.

## En français

`peakdemands` détecte les efforts (sprints, accélérations, décélérations, changements de
direction, séquences répétées), calcule une charge mécanique qui compte la direction
(travail lié à la vitesse selon Gray 2018, travail lié aux virages par axe selon Cavagna 1976,
signatures de mouvement), calcule les scénarios les plus exigeants par fenêtres
glissantes et leur loi de puissance, décompose l'accélération en vitesse et en direction,
et calcule le temps d'exposition (IET) proposé par Buchheit et al. (2026, « GPS 3.0 »).
Un rapport PDF de séance, en français ou en anglais, sort en une ligne.

## References

Buchheit M., Lopez Sagarra A., Boskovic A., Komino P., Norman D., Hader K. (2026). GPS 3.0. *Sport Perf Sci Rep* 280 ·
Buchheit & Simpson (2017). *IJSPP* 12(S2):S235 · Bundle, Hoyt & Weyand (2003). *J Appl Physiol* 95:1955 ·
Buchheit et al. (2010). *Int J Sports Med* 31:709 · Buchheit et al. (2023). *Biol Sport* 40:1057 ·
Spencer et al. (2004). *J Sports Sci* 22:843 · Harper, Carling & Kiely (2019). *Sports Med* 49:1923 ·
Varley, Elias & Aughey (2012). *IJSPP* 7:183 · Fereday et al. (2020). *J Sci Med Sport* 23:764 ·
Delaney et al. (2018). *Sci Med Football* 2:86 · Novak et al. (2021). *J Sports Sci* 39:1850 ·
Whitehead et al. (2018). *Sports Med* 48:2549 · Rico-González et al. (2022). *Biol Sport* 39:695 ·
Gray, Shorter, Cummins, Murphy & Waldron (2018). *Sports Med* 48:1357 · Cavagna, Thys & Zamboni (1976). *J Physiol* 262:639.

Data: Metrica Sports sample data (github.com/metrica-sports/sample-data).

## Licence

MIT. Clément Verdier.
