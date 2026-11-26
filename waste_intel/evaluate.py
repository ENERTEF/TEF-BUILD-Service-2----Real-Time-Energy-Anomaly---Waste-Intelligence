"""Walk-forward evaluation.

1. Baseline accuracy: every day of the test window is predicted from earlier days only;
   MAE and RMSE per 15-minute interval and per day, against the same interval a week
   earlier as reference.

2. Detection on synthetic events. There is no register of confirmed waste events yet, so
   known events are injected into a copy of each meter's real data and the detector is run
   walk-forward over it:

     ghost_load        +X kWh per interval 00-06 for 10 nights
     unoccupied_waste  closed-day load raised by half the open-day gap, 09-17, two weekends
     equipment_fault   +30% of the opening-hours load for 3 open days
     degradation_drift linear ramp to +15% of all load over 35 days

   Recall: share of injected events flagged with the right type (any-type also reported).
   Time to detection: from injection start to the moment the detection is confirmed
   (3rd night for ghost loads, 3rd hour for interval events, the day for drift), in hours.
   Precision: among flags the injection created (flags also present on the untouched copy
   are excluded, their truth is unknown), the share that overlap the injected event.
   False alarms: flags on the untouched copy per meter-week; real data may hold real waste,
   so this is an upper bound.
   Waste MAPE: injected kWh against the extra excess the detector measured on the injected
   days (drift: the drift event's estimate).

This measures the detector against a controlled truth; it does not replace validation on
confirmed real events, which the report lists as open.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .baseline import Baseline, naive_reference
from .config import CONFIG, SLOTS
from .detect import WASTE_TYPES, score_days
from .occupancy import is_open


def baseline_accuracy(b: Baseline, start: date, end: date) -> dict | None:
    a15, p15, r15, ad, pd_, rd = [], [], [], [], [], []
    d = start
    while d <= end:
        act = b.actual(d)
        base = b.expected(d)
        ref = naive_reference(b, d)
        if act is not None and base is not None and ref is not None:
            ok = np.isfinite(act) & np.isfinite(base["expected"]) & np.isfinite(ref)
            if ok.sum() >= CONFIG["day_min_slots"]:
                a15 += list(act[ok]); p15 += list(base["expected"][ok]); r15 += list(ref[ok])
                ad.append(float(np.sum(act[ok]))); pd_.append(float(np.sum(base["expected"][ok]))); rd.append(float(np.sum(ref[ok])))
        d += timedelta(days=1)
    if len(ad) < 7:
        return None
    a15, p15, r15, ad, pd_, rd = map(np.array, (a15, p15, r15, ad, pd_, rd))

    def m(a, p):
        return {"mae": round(float(np.mean(np.abs(a - p))), 4), "rmse": round(float(np.sqrt(np.mean((a - p) ** 2))), 4)}

    return {"days": len(ad), "interval": m(a15, p15), "interval_reference": m(a15, r15),
            "daily": m(ad, pd_), "daily_reference": m(ad, rd), "mean_daily_kwh": round(float(np.mean(ad)), 2)}


def _inject(m: pd.DataFrame, kind: str, start: date, school: bool) -> tuple[pd.DataFrame, dict] | None:
    m = m.copy()
    v = m.to_numpy(dtype=float)
    idx = {d: i for i, d in enumerate(m.index)}
    slots = np.arange(SLOTS)
    o0, o1 = CONFIG["open_hours_school"] if school else CONFIG["open_hours_municipal"]
    days, mask_by_day = [], {}
    mean_slot = float(np.nanmean(v))
    if kind == "ghost_load":
        night = slots < 24
        base_night = float(np.nanmedian(v[:, night]))
        amp = max(0.5 * base_night, 0.1 * mean_slot, 0.03)
        for i in range(10):
            d = start + timedelta(days=i)
            days.append(d); mask_by_day[d] = night * amp
    elif kind == "unoccupied_waste":
        open_rows = [i for d, i in idx.items() if is_open(d, school)]
        closed_rows = [i for d, i in idx.items() if not is_open(d, school)]
        day_win = (slots >= 36) & (slots < 68)
        gap = float(np.nanmean(v[open_rows][:, day_win]) - np.nanmean(v[closed_rows][:, day_win])) if open_rows and closed_rows else 0
        amp = max(0.5 * gap, 0.1 * mean_slot, 0.03)
        d = start
        while len(days) < 4 and d < start + timedelta(days=21):
            if d.weekday() >= 5:
                days.append(d); mask_by_day[d] = day_win * amp
            d += timedelta(days=1)
    elif kind == "equipment_fault":
        occ = (slots >= o0 * 4) & (slots < o1 * 4)
        d = start
        while len(days) < 3 and d < start + timedelta(days=14):
            if is_open(d, school) and d in idx:
                row = v[idx[d]]
                days.append(d); mask_by_day[d] = occ * max(0.3 * float(np.nanmean(row[occ])), 0.03)
            d += timedelta(days=1)
    elif kind == "degradation_drift":
        for i in range(35):
            d = start + timedelta(days=i)
            if d in idx:
                days.append(d); mask_by_day[d] = ("ramp", 0.15 * (i + 1) / 35)
    injected = 0.0
    for d, add in mask_by_day.items():
        if d not in idx:
            continue
        row = v[idx[d]]
        if isinstance(add, tuple):
            delta = np.nan_to_num(row) * add[1]
        else:
            delta = add.astype(float)
        delta = np.where(np.isfinite(row), delta, 0.0)
        v[idx[d]] = row + delta
        injected += float(delta.sum())
    if not days or injected <= 0:
        return None
    first_slot = 0
    if kind == "unoccupied_waste":
        first_slot = 36
    elif kind == "equipment_fault":
        first_slot = o0 * 4
    return pd.DataFrame(v, index=m.index, columns=m.columns), {
        "kind": kind, "start": pd.Timestamp(days[0]) + pd.Timedelta(minutes=15 * first_slot),
        "end": pd.Timestamp(days[-1]) + pd.Timedelta(hours=23, minutes=45), "kwh": injected}


def _overlaps(ev: dict, truth: dict) -> bool:
    s, e = pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"])
    return s <= truth["end"] and e >= truth["start"]


def synthetic_tests(e: dict, weather, end: date) -> list[dict]:
    """Inject each event type into the last ~5 weeks and score the copy walk-forward over the
    same window as an untouched copy. Only what the injection changed is credited or blamed:
    flags already present on the untouched copy are real-data findings of unknown truth."""
    m = e["matrix"]
    school = e["school"]
    win0 = end - timedelta(days=37)
    regd = lambda evs: [x for x in evs if x["type"] in WASTE_TYPES and x["confidence"] >= CONFIG["confidence_min"]]
    clean = score_days(Baseline(m, weather, school), win0, end, school, weather)
    clean_evs = regd(clean["events"])
    clean_keys = {(x["type"], x["start"][:10]) for x in clean_evs}
    clean_excess = {x["day"]: x.get("excess", 0.0) for x in clean["days"] if x.get("scored")}
    results = []
    plans = {"ghost_load": end - timedelta(days=24), "unoccupied_waste": end - timedelta(days=27),
             "equipment_fault": end - timedelta(days=20), "degradation_drift": end - timedelta(days=34)}
    for kind, start in plans.items():
        inj = _inject(m, kind, start, school)
        if inj is None:
            continue
        mm, truth = inj
        scored = score_days(Baseline(mm, weather, school), win0, end, school, weather)
        evs = [x for x in regd(scored["events"]) if (x["type"], x["start"][:10]) not in clean_keys]
        hits = [x for x in evs if _overlaps(x, truth)]
        right = [x for x in hits if x["type"] == kind or (kind == "unoccupied_waste" and x["type"] == "hvac_unoccupied")]
        ttd = None
        if right:
            confirm = min(pd.Timestamp(x.get("confirmed_at") or x["end"]) for x in right)
            ttd = max((confirm - truth["start"]).total_seconds() / 3600, 0.0)
        # waste estimate: what the injection added to the detected excess, day by day
        t0, t1 = truth["start"].date(), truth["end"].date()
        est = sum(max(x.get("excess", 0.0) - clean_excess.get(x["day"], 0.0), 0.0)
                  for x in scored["days"] if x.get("scored") and t0 <= date.fromisoformat(x["day"]) <= t1)
        if kind == "degradation_drift":
            est = sum(x["excess_kwh"] for x in right) or est
        results.append({"kind": kind, "detected": bool(right), "detected_any_type": bool(hits),
                        "flags": len(evs), "true_flags": len(hits),
                        "hours_to_detection": None if ttd is None else round(ttd, 1),
                        "injected_kwh": round(truth["kwh"], 1), "estimated_kwh": round(est, 1),
                        "ape": round(abs(est - truth["kwh"]) / truth["kwh"], 3) if right else None})
    results.append({"kind": "clean", "flags": len(clean_evs), "days": (end - win0).days + 1})
    return results


def aggregate(tests: list[list[dict]]) -> dict:
    flat = [t for ts in tests for t in ts]
    out = {}
    for kind in ["ghost_load", "unoccupied_waste", "equipment_fault", "degradation_drift"]:
        r = [t for t in flat if t["kind"] == kind]
        if not r:
            continue
        det = [t for t in r if t["detected"]]
        flags = sum(t["flags"] for t in r)
        true_flags = sum(t["true_flags"] for t in r)
        ttd = [t["hours_to_detection"] for t in det if t["hours_to_detection"] is not None]
        apes = [t["ape"] for t in det if t["ape"] is not None]
        out[kind] = {"injected": len(r), "recall": round(len(det) / len(r), 3),
                     "recall_any_type": round(sum(t["detected_any_type"] for t in r) / len(r), 3),
                     "precision": round(true_flags / flags, 3) if flags else None,
                     "median_hours_to_detection": round(float(np.median(ttd)), 1) if ttd else None,
                     "waste_mape": round(float(np.mean(apes)), 3) if apes else None}
    clean = [t for t in flat if t["kind"] == "clean"]
    if clean:
        meter_weeks = sum(t["days"] for t in clean) / 7
        out["clean"] = {"meters": len(clean), "flags": sum(t["flags"] for t in clean),
                        "flags_per_meter_week": round(sum(t["flags"] for t in clean) / meter_weeks, 3)}
    return out
