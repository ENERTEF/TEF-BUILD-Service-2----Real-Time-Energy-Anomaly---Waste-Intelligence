"""Per-building expected consumption.

For a target day the baseline is built from the building's own past days of the same type
(workday, Saturday, Sunday/holiday, and for schools weekday vacations), at most twelve
months back, choosing the days whose heating and cooling degree days are closest to the
target's, with age weighed in (14 days count as one degree day), so recent weeks are
preferred unless the weather has moved away from them. Per 15-minute interval it returns the median of
those days and a robust spread (scaled MAD), and the same for the daily total.

This is degree-day normalisation by matching rather than by regression: with three to
twelve months of history per meter a matched reference is more robust, and it can be
explained to an operator as "the eight most similar days this building has had".
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import CONFIG, SLOTS
from .occupancy import day_type

FALLBACK = {"vacation_weekday": ["saturday"], "saturday": ["sunday_holiday"],
            "sunday_holiday": ["saturday"], "workday": []}


class Baseline:
    def __init__(self, matrix: pd.DataFrame, weather: pd.DataFrame, school: bool):
        c = CONFIG
        self.m = matrix
        self.days = list(matrix.index)
        self.idx = {d: i for i, d in enumerate(self.days)}
        self.v = matrix.to_numpy(dtype=float)
        complete = np.isfinite(self.v).sum(axis=1) >= c["day_min_slots"]
        self.complete = complete
        self.school = school
        self.types = np.array([day_type(d, school) for d in self.days])
        w = weather.reindex(self.days)
        self.hdd = w["hdd"].to_numpy(dtype=float)
        self.cdd = w["cdd"].to_numpy(dtype=float)
        self.weather = weather
        # daily totals with missing intervals filled by the day's own mean slot, complete days only
        with np.errstate(invalid="ignore"):
            self.total = np.where(complete, np.nanmean(self.v, axis=1) * SLOTS, np.nan)
        self.ordinal = np.array([d.toordinal() for d in self.days])

    def weather_of(self, d: date) -> tuple[float, float]:
        if d in self.weather.index:
            r = self.weather.loc[d]
            return float(r["hdd"]), float(r["cdd"])
        return 0.0, 0.0

    def expected(self, d: date, exclude: set | None = None, lag_days: int = 1,
                 types: list[str] | None = None, skip_months: tuple = ()) -> dict | None:
        c = CONFIG
        target_type = day_type(d, self.school)
        hdd, cdd = self.weather_of(d)
        lo = d.toordinal() - c["baseline_lookback_days"]
        hi = d.toordinal() - lag_days
        base_ok = self.complete & (self.ordinal >= lo) & (self.ordinal <= hi) & np.isfinite(self.hdd)
        if exclude:
            mem = c["exclusion_memory_days"]
            base_ok &= np.array([not (x in exclude and (d - x).days <= mem) for x in self.days])
        if skip_months:
            base_ok &= np.array([x.month not in skip_months for x in self.days])
        for t in [target_type] + (types or FALLBACK[target_type]):
            cand = np.where(base_ok & (self.types == t))[0]
            if len(cand) >= c["baseline_min_neighbours"]:
                break
        else:
            return None
        age = d.toordinal() - self.ordinal[cand]
        dist = np.abs(self.hdd[cand] - hdd) + np.abs(self.cdd[cand] - cdd) + age / c["baseline_age_per_dd"]
        order = np.argsort(dist)[: c["baseline_neighbours"]]
        pick = cand[order]
        wgap = float(np.mean(np.abs(self.hdd[pick] - hdd) + np.abs(self.cdd[pick] - cdd)))
        rows = self.v[pick]
        exp = np.nanmedian(rows, axis=0)
        sig = 1.4826 * np.nanmedian(np.abs(rows - exp), axis=0)
        mean_slot = float(np.nanmean(exp)) if np.isfinite(exp).any() else 0.0
        sig = np.fmax(sig, np.fmax(0.10 * exp, 0.05 * mean_slot))
        sig = np.fmax(np.nan_to_num(sig, nan=0.01), 0.01)
        tot = self.total[pick]
        tot_med = float(np.nanmedian(tot))
        tot_sig = max(1.4826 * float(np.nanmedian(np.abs(tot - tot_med))), 0.05 * tot_med, 0.5)
        return {"expected": exp, "sigma": sig, "rows": rows, "total": tot_med, "total_sigma": tot_sig,
                "day_type": target_type, "type_used": t, "n": int(len(pick)), "weather_gap": round(wgap, 2),
                "neighbours": [str(self.days[i]) for i in pick], "hdd": hdd, "cdd": cdd}

    def actual(self, d: date) -> np.ndarray | None:
        i = self.idx.get(d)
        return None if i is None else self.v[i]

    def actual_total(self, d: date) -> float:
        i = self.idx.get(d)
        return float("nan") if i is None else float(self.total[i])


def naive_reference(b: Baseline, d: date) -> np.ndarray | None:
    """Same interval one week earlier: the reference a baseline must beat."""
    return b.actual(d - timedelta(days=7))
