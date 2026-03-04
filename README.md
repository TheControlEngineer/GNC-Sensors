
# GNC-Sensors 

- Status :- *In active developement* 
- Update :- *Validation ongoing against  OSIRIS-REx OLA* (03/03/2026)

  <img width="2053" height="830" alt="image" src="https://github.com/user-attachments/assets/f80edbc1-642f-45cb-ba3f-cdbc5c9d9cf5" />


## Sensor Model

The library first implements a configurable time-of-flight LiDAR model covering the full measurement chain from pulse emission to timestamped output. All model parameters are exposed through a single configuration class, allowing the same code to represent different instruments by changing config values.

**Measurement pipeline** (`lidar.py`, 761 lines):

Range noise (Gaussian floor + range-proportional) → bias random walk and drift → scale factor error → quantization → output saturation → dropout (range-dependent) → latency and jitter scheduling → platform-motion interpolation → output with optional truth metadata.

**Radiometric detection** (`physics.py`, 398 lines):

Pulse energy and beam divergence → Lambertian + retro-reflective surface model → 1/r² geometric falloff → beam footprint area → Beer-Lambert two-way atmospheric extinction → SNR-based soft detection threshold. Includes a ray-casting scene engine with sphere, plane, AABB, and triangle primitives for closed-loop scan simulation.

**Utilities** (`math_utils.py`, 213 lines):

3D vector operations, rotation matrices, spherical-to-Cartesian conversion, linear interpolation with extrapolation guards.

## Verification Suite

Statistical test suite using `pytest` with auto-generated HTML reports containing embedded diagnostic plots, structured test metadata (description, goal, passing criteria), and formal acceptance logic based on confidence intervals rather than arbitrary tolerances.

**Test modules** (~5,200 lines across 9 modules):

| Module | Lines | Coverage |
|---|---|---|
| `test_noise.py` | 857 | Gaussian noise statistics, bias random walk, deterministic drift, scale factor, outlier injection rate and magnitude |
| `test_range_accuracy.py` | 449 | Zero-noise boresight sweep, FoV boundary transitions, range gate boundaries |
| `test_beam_geometry.py` | 649 | Planar wall intersection, full-frame sphere consistency, multi-object occlusion, rotated sensor orientation |
| `test_dropout_quantization.py` | 583 | Dropout probability vs range with binomial CI, quantization staircase, saturation clipping vs invalidation |
| `test_power_detection.py` | 564 | 1/r⁴ power law, Beer-Lambert extinction, Lambertian cosine, retro-reflectivity, SNR threshold, empty scene |
| `test_timing.py` | 699 | Sampling rate clock, latency pipeline, jitter injection, time reversal reset, position interpolation |
| `test_ola_specsheet.py` | 160 | OLA-LELT spec-sheet reproduction (see below) |
| `test_ola_orbit_a_phase.py` | 915 | Orbit A flight data cross-validation (see below) |
| `test_ola_crossval.py` | 329 | Orbit B flight data cross-validation (see below) |

Noise and rate tests use chi-squared confidence intervals for variance validation and binomial confidence intervals for rate validation.

### Running the tests

```bash
pip install pytest pytest-html pytest-metadata matplotlib numpy
```

```bash
# Run core verification suite with HTML report
pytest tests/test_lidar/ \
  --ignore=tests/test_lidar/test_ola_crossval.py \
  --ignore=tests/test_lidar/test_ola_orbit_a_phase.py \
  --html=report.html --self-contained-html

# Run OLA spec-sheet validation only (no data download required)
pytest tests/test_lidar/test_ola_specsheet.py --html=report_ola_specsheet.html --self-contained-html
```

## Flight Data Validation

The model is validated against the OSIRIS-REx Laser Altimeter (OLA), a scanning LiDAR that operated at asteroid Bennu from 2018–2020. Validation proceeds at two levels.

### Level 1 — Spec-sheet reproduction

The model is configured with published OLA LELT parameters (Daly et al. 2017) and verified against the instrument's performance envelope over 10,000 Monte Carlo samples per test range:

- Range noise σ consistent with published 0.06 m precision at 99% confidence (chi-squared test)
- Detection rate > 95% at 700 m on a 4.4% albedo target (Bennu-like)
- Range gating correctly rejects outside the 500–1200 m operational envelope

No flight data required. Configuration in `ola_config.py`.

### Level 2 — Flight data cross-validation

Model output statistics are compared against calibrated OLA Level 2 data from the NASA Planetary Data System. The test loads PDS4 binary tables, reconstructs geometric input ranges from body-fixed and spacecraft position columns, derives instrument noise and dropout profiles from the data itself, configures the model to match, replays the same input ranges through the model, and compares per-range-bin statistics.

**Orbit A validation** (`test_ola_orbit_a_phase.py`): Self-calibrating pipeline that extracts noise profile σ(r), dropout profile P(r), bias, and scale factor from flight residuals, then verifies the configured model reproduces these statistics. Passing criteria: detection MAE ≤ 1%, σ relative error median ≤ 15% (p95 ≤ 25%, p99 ≤ 30%).

**Orbit B validation** (`test_ola_crossval.py`): Compares LELT-configured model against ~700 m range Orbital B data using windowed precision estimates and binomial detection fraction tests.

To run flight data tests, download one OLA L2 file from the PDS archive and place it in the test data directory:

```bash
mkdir -p tests/test_lidar/data/ola_orbit_b
cd tests/test_lidar/data/ola_orbit_b
wget https://sbnarchive.psi.edu/pds4/orex/orex.ola/data_calibrated/orbit_b/20190701_ola_scil2id04000.dat
wget https://sbnarchive.psi.edu/pds4/orex/orex.ola/data_calibrated/orbit_b/20190701_ola_scil2id04000.xml
```

```bash
pytest tests/test_lidar/test_ola_crossval.py --html=report_ola_crossval.html --self-contained-html
pytest tests/test_lidar/test_ola_orbit_a_phase.py --html=report_ola_orbit_a_phase.html --self-contained-html
```

Flight data files are gitignored. The tests skip gracefully when data is not present.

## Repository Structure

```
GNC-Sensors/
├── sensors/
│   ├── Config.py              Configuration dataclass
│   ├── lidar.py               Sensor model (measurement pipeline + radiometric detection)
│   ├── physics.py             Ray-casting scene engine (sphere, plane, AABB, triangle)
│   └── math_utils.py          Vector/rotation utilities
├── tests/
│   └── test_lidar/
│       ├── conftest.py        HTML report configuration with custom columns and plot embedding
│       ├── helpers.py         Chi-squared bounds, binomial CI, plot attachment utilities
│       ├── ola_config.py      OLA-LELT instrument configuration
│       ├── pds4_parser.py     PDS4 binary table parser for OLA flight data
│       ├── extract_ola_statistics.py   Windowed precision and binned statistics
│       ├── test_noise.py
│       ├── test_range_accuracy.py
│       ├── test_beam_geometry.py
│       ├── test_dropout_quantization.py
│       ├── test_power_detection.py
│       ├── test_timing.py
│       ├── test_ola_specsheet.py
│       ├── test_ola_crossval.py
│       ├── test_ola_orbit_a_phase.py
│       └── data/ report             Flight data (gitignored)
|         
└── README.md
```

## References

Daly, M. G. et al. (2017). The OSIRIS-REx Laser Altimeter (OLA) Investigation and Instrument. *Space Science Reviews*, 212(1-2), 899–924. [doi:10.1007/s11214-017-0375-3](https://doi.org/10.1007/s11214-017-0375-3)

OLA flight data: NASA Planetary Data System, `urn:nasa:pds:orex.ola`. [PDS Archive](https://arcnav.psi.edu/urn:nasa:pds:orex.ola)



