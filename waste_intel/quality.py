"""Data quality control.

Two jobs: decide which meters are fit for detection (with the reason when not), and turn
data faults inside the monitoring window into 'meter_quality' detections, so an operator
sees a broken meter as a broken meter and not as waste.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import CONFIG, SLOTS


def full_matrix(m: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    return m.reindex(index=days, columns=range(SLOTS))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of consecutive True runs."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def _spikes(v: np.ndarray) -> np.ndarray:
    p99 = np.nanquantile(v, 0.99) if np.isfinite(v).any() else 0
    prev = np.concatenate([[np.nan], v[:-1]])
    nxt = np.concatenate([v[1:], [np.nan]])
    nb = np.fmax(np.nan_to_num(prev), np.nan_to_num(nxt))
    return (v > 4 * nb) & (v > 2 * p99) & (v > 0.05)


def _stuck(v: np.ndarray, min_len: int = 16) -> np.ndarray:
    out = np.zeros(len(v), bool)
    eq = np.concatenate([[False], (v[1:] == v[:-1]) & (v[1:] > 0)])
    for s, l in _runs(eq):
        if l + 1 >= min_len:
            out[max(s - 1, 0): s + l] = True
    return out


def assess(e: dict, run_date: date, ref_date: date) -> dict:
    """run_date: last day analysed for this meter; ref_date: the date freshness is judged
    against (today for live feeds, the archive's end for historical DEI telemetry)."""
    c = CONFIG
    m = e["matrix"]
    last = max(m.index) if len(m) else None
    start = run_date - timedelta(days=c["qc_window_days"] - 1)
    w = full_matrix(m, start, run_date) if last else None
    reasons = []
    metrics = {"last_day": str(last) if last else None, "history_days": int(m.notna().any(axis=1).sum())}
    if last is None or (ref_date - last).days > c["qc_max_age_days"]:
        reasons.append(f"χωρίς δεδομένα μετά τις {last}" if last else "χωρίς δεδομένα")
        return {"ok": False, "reasons": reasons, "metrics": metrics}
    v = w.to_numpy().ravel()
    present = np.isfinite(v)
    vals = v[present]
    cov = present.mean()
    zero = float((vals == 0).mean()) if len(vals) else 1.0
    spike = float(_spikes(v)[present].mean()) if len(vals) else 0.0
    stuck = float(_stuck(v)[present].mean()) if len(vals) else 0.0
    kwh_day = float(np.nanmean(v) * SLOTS) if len(vals) else 0.0
    neg = int((vals < 0).sum())
    metrics.update(coverage=round(cov, 3), zero_share=round(zero, 3), spike_share=round(spike, 4),
                   stuck_share=round(stuck, 4), kwh_day=round(kwh_day, 1), negative=neg)
    if cov < c["qc_min_coverage"]:
        reasons.append(f"κάλυψη {cov:.0%} (< {c['qc_min_coverage']:.0%})")
    if zero > c["qc_max_zero_share"]:
        reasons.append(f"μηδενικά {zero:.0%}")
    if spike > c["qc_max_spike_share"]:
        reasons.append(f"ακραίες αιχμές {spike:.1%}")
    if stuck > c["qc_max_stuck_share"]:
        reasons.append(f"κολλημένη τιμή {stuck:.1%}")
    if neg:
        reasons.append(f"{neg} αρνητικές τιμές")
    if kwh_day < c["qc_min_kwh_day"]:
        reasons.append(f"{kwh_day:.1f} kWh/ημέρα (πολύ χαμηλή)")
    if e["school"] and kwh_day > c["qc_max_kwh_day_school"]:
        reasons.append(f"{kwh_day:,.0f} kWh/ημέρα (μη ρεαλιστική για σχολείο)")
    metrics["fingerprint"] = round(float(np.nansum(v)), 3)
    return {"ok": not reasons, "reasons": reasons, "metrics": metrics}


def drop_duplicates(entities: list[dict]) -> None:
    """Two schools can share one physical meter, and some feeds repeat a series under
    two ids. Keep the first; the others are excluded with a pointer to it."""
    seen = {}
    for e in entities:
        q = e["qc"]
        if not q["ok"]:
            continue
        key = (e["source"], q["metrics"].get("fingerprint"))
        if key in seen:
            q["ok"] = False
            q["reasons"].append(f"ίδια σειρά με {seen[key]}")
        else:
            seen[key] = e["id"]


def quality_events(e: dict, start: date, end: date, outage_days: set | None = None) -> list[dict]:
    """Gaps, stuck values, spikes and dead stretches inside the monitoring window. Days on
    which the whole source went quiet (our collection was down) are not blamed on the meter."""
    w = full_matrix(e["matrix"], start, end)
    v = w.to_numpy().ravel()
    days = list(w.index)
    if outage_days:
        keep = np.repeat([d not in outage_days for d in days], SLOTS)
        mask_outage = ~keep
    else:
        mask_outage = np.zeros(len(v), bool)
    out = []

    def ts(i):
        d = days[i // SLOTS]
        return f"{d}T{(i % SLOTS) // 4:02d}:{(i % 4) * 15:02d}"

    for s, l in _runs(~np.isfinite(v) & ~mask_outage):
        if l >= 16:
            out.append({"kind": "gap", "start": ts(s), "end": ts(min(s + l, len(v)) - 1), "slots": l,
                        "detail": f"κενό {l / 4:.1f} ωρών"})
    for s, l in _runs(_stuck(v)):
        out.append({"kind": "stuck", "start": ts(s), "end": ts(s + l - 1), "slots": l,
                    "detail": f"ίδια τιμή για {l / 4:.1f} ώρες"})
    sp = np.where(_spikes(v))[0]
    if len(sp):
        out.append({"kind": "spike", "start": ts(sp[0]), "end": ts(sp[-1]), "slots": int(len(sp)),
                    "detail": f"{len(sp)} απομονωμένες αιχμές"})
    typical = np.nanmedian(v)
    if typical > 0:
        for s, l in _runs(np.nan_to_num(v, nan=-1) == 0):
            if l >= 24:
                out.append({"kind": "zero_run", "start": ts(s), "end": ts(s + l - 1), "slots": l,
                            "detail": f"μηδενική ένδειξη για {l / 4:.1f} ώρες"})
    return out
