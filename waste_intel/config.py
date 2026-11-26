"""Every threshold the service uses. A copy is written with each run, so any past
detection can be reconstructed with the values that produced it."""

import os
from pathlib import Path

MODEL_VERSION = "waste-intel 1.0.0"

OUT_DIR = Path(os.getenv("WASTE_INTEL_DIR", "outputs"))

SLOTS = 96                      # 15-minute intervals per day
TZ = "Europe/Athens"

CONFIG = {
    # Data quality: a meter enters detection only if its last QC_WINDOW_DAYS pass all of these
    "qc_window_days": 60,
    "qc_max_age_days": 7,           # last reading no older than this before the run date
    "qc_min_coverage": 0.85,        # share of expected 15-minute readings present
    "qc_max_zero_share": 0.10,
    "qc_max_spike_share": 0.005,    # isolated spikes: > 4x both neighbours and > 2x p99
    "qc_max_stuck_share": 0.01,     # same non-zero value for 4 hours or more
    "qc_min_kwh_day": 5.0,
    "qc_max_kwh_day_school": 1500.0,  # above this a school meter is not physically plausible
    "day_min_slots": 80,            # a day enters the baseline only with 80 of 96 readings

    # Baseline: nearest days of the same type, matched on heating and cooling degree days
    "baseline_lookback_days": 365,
    "baseline_neighbours": 8,
    "baseline_min_neighbours": 4,
    "baseline_age_per_dd": 14,      # 14 days of age weigh as much as 1 degree day of weather difference
    "exclusion_memory_days": 14,    # flagged days stay out of baselines for 14 days; a change that
                                    # persists longer becomes the new normal (and drift watches it)

    # Interval scoring
    "z_threshold": 3.0,
    "min_excess_share": 0.15,       # excess must also be 15% of the expected value...
    "min_excess_kwh": 0.02,         # ...and at least this many kWh per 15 minutes
    "block_z": 3.0,                 # hour-block test: a block's sum against the same block on the
    "block_min_rise": 0.10,         # matched days, > 3 robust sigmas and > 10% above their median
    "night_hours": (0, 6),          # unoccupied every day of the year
    "open_hours_school": (8, 16),
    "open_hours_municipal": (7, 17),

    # Events
    "ghost_min_nights": 3,          # consecutive nights with elevated base load
    "event_min_slots": 12,          # 3 hours of consecutive anomalous intervals...
    "event_min_kwh": 1.0,           # ...carrying at least 1 kWh and 3% of the expected day
    "event_min_day_share": 0.03,
    "post_holiday_min_closed_days": 3,
    "post_holiday_z": 2.5,          # on the daily total
    "drift_window_days": 28,
    "drift_min_rise": 0.07,         # 7% above the reference level on average over 28 days...
    "drift_cusum_h": 4.0,           # ...confirmed by a one-sided CUSUM in robust-sigma units
    "drift_cusum_k": 0.25,          # CUSUM slack: tuned to sustained shifts of about half a sigma
    "max_weather_gap": 1.5,         # degree days: matched days further than this from the target's
                                    # weather are not a fair reference (drift skips the day, other
                                    # detections lose confidence)
    "drift_reference_lag_days": 56, # reference built from days at least this old
    "confidence_min": 0.7,          # below this a detection is reported but not registered
    "single_day_factor": 0.8,       # one-day episodes are scaled down; persistence over 3+ days restores it

    # Waste register
    "monitor_days": 28,             # waste rate measured on the last 28 days...
    "post_holiday_per_year": 8,     # ...post-holiday spikes are scaled by occurrences instead
    "trend_change": 0.20,           # +-20% versus the previous 28 days

    # Cost
    "price_min": 0.08, "price_max": 0.60,   # €/kWh sanity band for prices from bills
    "price_years": (2023, 2024),
}
