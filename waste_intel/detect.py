"""Anomaly scoring and waste quantification.

Walk-forward: each day is scored against a baseline built only from days before it, and
days that produced a detection are kept out of later baselines, so a persisting fault does
not become the new normal after a week.

An interval is anomalous when it exceeds the baseline by more than z_threshold robust
sigmas AND by a practical margin (15% of expected and 0.02 kWh). Anomalous intervals are
grouped into typed events:

  ghost_load           elevated night base load (00-06) on >= 3 consecutive nights
  unoccupied_waste     >= 2 h of excess while the building is closed (outside opening hours,
                       weekends, holidays, school vacations)
  hvac_unoccupied      the same, on a heating or cooling day, with indoor temperature showing
                       the space was conditioned (only where a thermostat exists)
  equipment_fault      >= 2 h of excess during opening hours
  post_holiday_spike   first open day after >= 3 closed days, daily total > 2.5 sigma
  degradation_drift    daily consumption >= 10% above a reference at least 8 weeks old,
                       confirmed by a one-sided CUSUM
  meter_quality        gaps, stuck values, spikes, dead stretches (from quality.py)
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np

from .baseline import Baseline
from .config import CONFIG, MODEL_VERSION, SLOTS
from .occupancy import is_open

WASTE_TYPES = ["ghost_load", "unoccupied_waste", "hvac_unoccupied", "equipment_fault",
               "post_holiday_spike", "degradation_drift"]


def _slot_time(d: date, s: int) -> str:
    return f"{d}T{s // 4:02d}:{(s % 4) * 15:02d}"


def _runs(mask: np.ndarray, allow_holes: int = 1) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j, holes = i, 0
        while j < n and (mask[j] or holes < allow_holes and j + 1 < n and mask[j + 1]):
            holes = holes + 1 if not mask[j] else 0
            j += 1
        out.append((i, j - i))
        i = j
    return out


def _block(act: np.ndarray, rows: np.ndarray, mask: np.ndarray) -> tuple[float, float, float] | None:
    """Sum of a block of hours against the same block on the matched days: (z, excess, median)."""
    if mask.sum() == 0 or np.isfinite(act[mask]).mean() < 0.8:
        return None
    x = float(np.nanmean(act[mask]) * mask.sum())
    ref = np.nanmean(rows[:, mask], axis=1) * mask.sum()
    ref = ref[np.isfinite(ref)]
    if len(ref) < CONFIG["baseline_min_neighbours"]:
        return None
    med = float(np.median(ref))
    sig = max(1.4826 * float(np.median(np.abs(ref - med))), 0.05 * med, 0.05)
    return (x - med) / sig, x - med, med


def _confidence(mean_z: float, duration: float, consistency: float, quality: float) -> float:
    strength = min(max(mean_z / (2 * CONFIG["z_threshold"]), 0.0), 1.0)
    c = 0.35 * strength + 0.25 * min(max(duration, 0.0), 1.0) + 0.2 * consistency + 0.2 * quality
    return round(min(max(c, 0.0), 1.0), 3)


def _hours(school: bool) -> tuple[int, int]:
    return CONFIG["open_hours_school"] if school else CONFIG["open_hours_municipal"]


def score_days(b: Baseline, start: date, end: date, school: bool, weather, indoor: dict | None = None) -> dict:
    """Walk-forward over [start, end]. Returns per-day scoring and typed events."""
    c = CONFIG
    n0, n1 = c["night_hours"]
    o0, o1 = _hours(school)
    slots = np.arange(SLOTS)
    night = (slots >= n0 * 4) & (slots < n1 * 4)
    flagged: set = set()
    days_out = []
    events = []
    ghost_run: list = []
    closed_streak = 0
    d = start
    while d <= end:
        act = b.actual(d)
        base = b.expected(d, exclude=flagged)
        open_day = is_open(d, school)
        if act is None or base is None or np.isfinite(act).sum() < SLOTS // 2:
            days_out.append({"day": str(d), "scored": False})
            if ghost_run:
                _close_ghost(ghost_run, events, b)
                ghost_run = []
            closed_streak = closed_streak + 1 if not open_day else 0
            d += timedelta(days=1)
            continue
        exp, sig = base["expected"], base["sigma"]
        excess = act - exp
        z = np.where(np.isfinite(act), excess / sig, 0.0)
        margin = np.fmax(c["min_excess_share"] * exp, c["min_excess_kwh"])
        anom = np.isfinite(act) & (z > c["z_threshold"]) & (excess > margin)
        occupied = open_day & (slots >= o0 * 4) & (slots < o1 * 4)
        quality = float(np.isfinite(act).mean()) * min(base["n"] / c["baseline_neighbours"], 1.0)             / (1.0 + max(base["weather_gap"] - c["max_weather_gap"], 0.0))
        wctx = _weather_ctx(weather, d)
        ctx = {"day_type": base["day_type"], "open": open_day, "weather": wctx, "weather_gap_dd": base["weather_gap"],
               "baseline_days": base["neighbours"], "baseline_type": base["type_used"]}
        if indoor is not None and d in indoor:
            ctx["indoor_temp"] = round(indoor[d], 1)
        day_flag = False

        # ghost load: most of the night anomalous
        night_anom = anom & night
        nb = _block(act, base["rows"], night)
        night_block = nb is not None and nb[0] > c["block_z"] and nb[1] > c["block_min_rise"] * nb[2] and nb[1] > c["min_excess_kwh"] * night.sum()
        if night_anom.sum() >= 0.5 * night.sum() or night_block:
            if night_block and night_anom.sum() < 0.5 * night.sum():
                night_anom = night & np.isfinite(act)
                excess_n = np.full(SLOTS, 0.0); excess_n[night] = nb[1] / night.sum()
            else:
                excess_n = excess
            ghost_run.append({"day": d, "excess": float(np.nansum(np.where(night_anom, excess_n, 0))),
                              "obs": float(np.nansum(act[night])), "exp": float(np.nansum(exp[night])),
                              "z": max(float(np.mean(z[night_anom])), nb[0] if nb else 0.0),
                              "cons": float((anom & night).sum() / night.sum()) if not night_block else max(float((anom & night).sum() / night.sum()), 0.7),
                              "q": quality, "ctx": ctx})
        elif ghost_run:
            day_flag |= _close_ghost(ghost_run, events, b, flagged)
            ghost_run = []

        # interval events outside the night window
        for mask, kind in ((anom & ~night & ~occupied, "unoccupied"), (anom & occupied, "occupied")):
            block_mask = (~night & ~occupied) if kind == "unoccupied" else occupied
            found = False
            for s, l in _runs(mask, allow_holes=2):
                if l < c["event_min_slots"]:
                    continue
                seg = slice(s, s + l)
                hit = mask[seg]
                ex = float(np.nansum(np.where(hit, excess[seg], 0)))
                if ex < max(c["event_min_kwh"], c["event_min_day_share"] * base["total"]):
                    continue
                typ = "equipment_fault" if kind == "occupied" else _unoccupied_type(wctx, ctx, indoor, d)
                events.append({
                    "type": typ, "start": _slot_time(d, s), "end": _slot_time(d, s + l - 1), "days": [str(d)],
                    "confirmed_at": _slot_time(d, s + c["event_min_slots"] - 1),
                    "excess_kwh": round(ex, 2), "observed_kwh": round(float(np.nansum(act[seg])), 2),
                    "expected_kwh": round(float(np.nansum(exp[seg])), 2),
                    "confidence": _confidence(float(np.mean(z[seg][hit])), l / (4 * c["event_min_slots"]),
                                              float(hit.mean()), quality),
                    "evidence": ctx,
                })
                day_flag = found = True
            if not found:
                bk = _block(act, base["rows"], block_mask)
                if bk and bk[0] > c["block_z"] and bk[1] > c["block_min_rise"] * bk[2] and                         bk[1] >= max(c["event_min_kwh"], c["event_min_day_share"] * base["total"]):
                    idx = np.where(block_mask)[0]
                    typ = "equipment_fault" if kind == "occupied" else _unoccupied_type(wctx, ctx, indoor, d)
                    events.append({
                        "type": typ, "start": _slot_time(d, int(idx[0])), "end": _slot_time(d, int(idx[-1])),
                        "days": [str(d)], "confirmed_at": _slot_time(d, int(idx[-1])),
                        "excess_kwh": round(bk[1], 2), "observed_kwh": round(bk[1] + bk[2], 2),
                        "expected_kwh": round(bk[2], 2),
                        "confidence": _confidence(bk[0], 0.5, 0.7, quality),
                        "evidence": ctx | {"test": "hour block"},
                    })
                    day_flag = True

        # post-holiday spike, on the daily total
        tot = b.actual_total(d)
        if open_day and closed_streak >= c["post_holiday_min_closed_days"] and math.isfinite(tot):
            zd = (tot - base["total"]) / base["total_sigma"]
            if zd > c["post_holiday_z"]:
                events.append({
                    "type": "post_holiday_spike", "start": _slot_time(d, 0), "end": _slot_time(d, SLOTS - 1),
                    "confirmed_at": _slot_time(d, SLOTS - 1),
                    "days": [str(d)], "excess_kwh": round(tot - base["total"], 2), "observed_kwh": round(tot, 2),
                    "expected_kwh": round(base["total"], 2),
                    "confidence": _confidence(zd, 0.8, 1.0, quality),
                    "evidence": ctx | {"closed_days_before": closed_streak},
                })
                day_flag = True
        closed_streak = 0 if open_day else closed_streak + 1

        if day_flag:
            flagged.add(d)
        days_out.append({"day": str(d), "scored": True, "observed": round(float(np.nansum(act)), 2),
                         "expected": round(float(np.nansum(exp)), 2),
                         "excess": round(float(np.nansum(np.where(anom, excess, 0))), 2),
                         "anomalous_slots": int(anom.sum()), "open": open_day})
        d += timedelta(days=1)
    if ghost_run:
        _close_ghost(ghost_run, events, b, flagged)

    events = merge_events(events)
    drift = _drift(b, start, end, school)
    if drift:
        events.append(drift)
    return {"days": days_out, "events": events, "flagged": sorted(str(x) for x in flagged)}


def merge_events(events: list[dict]) -> list[dict]:
    """One event per type and episode: same-type events on the same or consecutive days
    are one persisting problem, not several."""
    merged: list[dict] = []
    for ev in sorted(events, key=lambda x: (x["type"], x["start"])):
        last = merged[-1] if merged else None
        if (last and last["type"] == ev["type"] and ev["type"] in ("unoccupied_waste", "hvac_unoccupied", "equipment_fault")
                and (date.fromisoformat(ev["start"][:10]) - date.fromisoformat(last["end"][:10])).days <= 1):
            last["end"] = ev["end"]
            last["days"] = sorted(set(last["days"]) | set(ev["days"]))
            for k in ("excess_kwh", "observed_kwh", "expected_kwh"):
                last[k] = round((last[k] or 0) + (ev[k] or 0), 2)
            last["_conf"].append(ev["confidence"])
            last["evidence"] = ev["evidence"]
        else:
            merged.append(dict(ev, _conf=[ev["confidence"]]))
    f = CONFIG["single_day_factor"]
    for ev in merged:
        if ev["type"] in ("unoccupied_waste", "hvac_unoccupied", "equipment_fault"):
            n = len(ev["days"])
            base = sum(ev["_conf"]) / len(ev["_conf"])
            ev["confidence"] = round(min(1.0, base * (f + (1 - f) * min(n, 3) / 3) + 0.03 * max(n - 3, 0)), 3)
        ev.pop("_conf", None)
    return sorted(merged, key=lambda x: x["start"])


def _close_ghost(run: list, events: list, b: Baseline, flagged: set | None = None) -> bool:
    c = CONFIG
    if len(run) < c["ghost_min_nights"]:
        return False
    ex = sum(r["excess"] for r in run)
    events.append({
        "type": "ghost_load", "start": _slot_time(run[0]["day"], c["night_hours"][0] * 4),
        "end": _slot_time(run[-1]["day"], c["night_hours"][1] * 4 - 1),
        "days": [str(r["day"]) for r in run], "excess_kwh": round(ex, 2),
        "confirmed_at": _slot_time(run[c["ghost_min_nights"] - 1]["day"], c["night_hours"][1] * 4 - 1),
        "observed_kwh": round(sum(r["obs"] for r in run), 2), "expected_kwh": round(sum(r["exp"] for r in run), 2),
        "confidence": _confidence(float(np.mean([r["z"] for r in run])), len(run) / 7,
                                  float(np.mean([r["cons"] for r in run])), float(np.mean([r["q"] for r in run]))),
        "evidence": run[-1]["ctx"] | {"nights": len(run),
                                      "night_base_kw": round(run[-1]["obs"] / (c["night_hours"][1] - c["night_hours"][0]), 2)},
    })
    if flagged is not None:
        flagged.update(r["day"] for r in run)
    return True


def _unoccupied_type(wctx: dict, ctx: dict, indoor: dict | None, d: date) -> str:
    conditioned_day = (wctx.get("hdd") or 0) > 2 or (wctx.get("cdd") or 0) > 1
    if not conditioned_day or not indoor or d not in indoor:
        return "unoccupied_waste"
    # heating: indoor well above outdoor; cooling: indoor well below outdoor
    t_out = wctx.get("t_mean")
    if t_out is None:
        return "unoccupied_waste"
    if (wctx.get("hdd") or 0) > 2 and indoor[d] - t_out > 8:
        return "hvac_unoccupied"
    if (wctx.get("cdd") or 0) > 1 and t_out - indoor[d] > 3:
        return "hvac_unoccupied"
    return "unoccupied_waste"


def _weather_ctx(weather, d: date) -> dict:
    if d not in weather.index:
        return {}
    r = weather.loc[d]
    return {"t_mean": round(float(r["t_mean"]), 1), "hdd": round(float(r["hdd"]), 1),
            "cdd": round(float(r["cdd"]), 1), "climatology": bool(r["clim"])}


def _drift(b: Baseline, start: date, end: date, school: bool = False) -> dict | None:
    """Daily consumption against a reference at least eight weeks old, with a CUSUM so a
    slow rise that no single day would flag still accumulates into a detection."""
    c = CONFIG
    lag = c["drift_reference_lag_days"]
    d = end - timedelta(days=90)
    ratios = []
    while d <= end:
        tot = b.actual_total(d)
        if math.isfinite(tot):
            # August is the municipal holiday month; schools have it as their own day type already
            ref = b.expected(d, lag_days=lag, skip_months=() if school else (8,))
            if ref and ref["total"] > 0 and ref["weather_gap"] <= c["max_weather_gap"]:
                ratios.append((d, tot / ref["total"] - 1, tot - ref["total"]))
        d += timedelta(days=1)
    if len(ratios) < 42:
        return None
    r = np.array([x[1] for x in ratios])
    sigma = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 0.03)
    s, s_start, best = 0.0, 0, None
    for i, x in enumerate(r):
        new = max(0.0, s + x / sigma - c["drift_cusum_k"])
        if s == 0 and new > 0:
            s_start = i
        s = new
    recent = [x for x in ratios if x[0] > end - timedelta(days=c["drift_window_days"])]
    if not recent:
        return None
    rise = float(np.mean([x[1] for x in recent]))
    if s > c["drift_cusum_h"] and rise >= c["drift_min_rise"] and len(recent) >= 14:
        first = ratios[s_start][0]
        ex = float(sum(max(x[2], 0) for x in recent))
        best = {
            "type": "degradation_drift", "start": f"{first}T00:00", "end": f"{end}T23:45",
            "days": [str(x[0]) for x in recent], "excess_kwh": round(ex, 2), "confirmed_at": f"{end}T23:45",
            "observed_kwh": None, "expected_kwh": None,
            "confidence": _confidence(rise / sigma, (end - first).days / 28,
                                      float(np.mean([x[1] > 0 for x in recent])), 1.0),
            "evidence": {"mean_rise": round(rise, 3), "cusum": round(s, 2), "reference_lag_days": lag},
        }
    return best


def summarise(e: dict, scored: dict, quality_events: list, end: date) -> dict:
    """Waste register entries, status and trend for one meter."""
    c = CONFIG
    mon = c["monitor_days"]
    cur0 = end - timedelta(days=mon - 1)
    prev0 = cur0 - timedelta(days=mon)
    price = e["price"]
    detections = []
    for ev in scored["events"]:
        ev = dict(ev)
        ev["registered"] = ev["confidence"] >= c["confidence_min"]
        ev["model_version"] = MODEL_VERSION
        detections.append(ev)
    for q in quality_events:
        detections.append({"type": "meter_quality", "subtype": q["kind"], "start": q["start"], "end": q["end"],
                           "excess_kwh": 0.0, "confidence": 1.0, "registered": False,
                           "evidence": {"detail": q["detail"]}, "model_version": MODEL_VERSION})

    def in_period(ev, a, z):
        last = date.fromisoformat(ev["end"][:10])
        return a <= last <= z

    register = []
    for typ in WASTE_TYPES:
        evs = [x for x in detections if x["type"] == typ and x["registered"] and in_period(x, cur0, end)]
        if not evs:
            continue
        kwh = sum(x["excess_kwh"] for x in evs)
        if typ == "post_holiday_spike":
            annual = kwh / len(evs) * c["post_holiday_per_year"]
        elif typ == "degradation_drift":
            annual = kwh / max(len(evs[0]["days"]), 1) * 365
        else:
            annual = kwh / mon * 365
        register.append({"type": typ, "events": len(evs), "observed_excess_kwh": round(kwh, 1),
                         "annual_kwh": round(annual), "annual_eur": round(annual * price),
                         "confidence": round(max(x["confidence"] for x in evs), 2)})

    def excess_between(a, z):
        return sum(x["excess_kwh"] for x in detections
                   if x["type"] in WASTE_TYPES and x["type"] != "degradation_drift" and x["registered"] and in_period(x, a, z))

    cur, prev = excess_between(cur0, end), excess_between(prev0, cur0 - timedelta(days=1))
    if cur < 1 and prev < 1:
        trend = "stable"
    elif cur > prev * (1 + c["trend_change"]):
        trend = "deteriorating"
    elif cur < prev * (1 - c["trend_change"]):
        trend = "improving"
    else:
        trend = "stable"
    recent = end - timedelta(days=6)
    active = [x for x in detections if x["type"] in WASTE_TYPES and x["registered"] and in_period(x, recent, end)]
    dq_recent = [x for x in detections if x["type"] == "meter_quality" and in_period(x, recent, end)]
    if active:
        status = "anomalous"
    elif any(x["type"] in WASTE_TYPES and in_period(x, cur0, end) for x in detections):
        status = "watch"
    elif dq_recent:
        status = "data_issue"
    else:
        status = "normal"
    scored_days = [x for x in scored["days"] if x.get("scored") and date.fromisoformat(x["day"]) >= cur0]
    return {
        "status": status, "trend": trend, "register": register,
        "annual_kwh": sum(r["annual_kwh"] for r in register), "annual_eur": sum(r["annual_eur"] for r in register),
        "excess_current_kwh": round(cur, 1), "excess_previous_kwh": round(prev, 1),
        "observed_kwh": round(sum(x["observed"] for x in scored_days), 1),
        "expected_kwh": round(sum(x["expected"] for x in scored_days), 1),
        "detections": detections,
    }
