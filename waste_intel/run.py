"""Orchestration: one detection cycle over every meter, with a progress log, versioned
outputs and a markdown report.

  from waste_intel.run import run
  run(my_data_source, skip_eval=False, only=("sensor", "eunice", "dei"))

`my_data_source` is any implementation of `sources.DataSource`.

Outputs in WASTE_INTEL_DIR/<run_id>/ and a copy of the light files in WASTE_INTEL_DIR/latest/:
  config.json       thresholds, model version, run date, data end per source
  quality.json      every candidate meter: metrics, included or the reasons it was excluded
  buildings.json    per meter: status, trend, waste register lines, accuracy, evidence
  register.json     the portfolio waste register
  evaluation.json   baseline accuracy and synthetic-event detection results
  report.md         the human-readable report
Progress: service2.log and status.json next to the run folders.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np

from .baseline import Baseline
from .config import CONFIG, MODEL_VERSION, OUT_DIR
from .detect import WASTE_TYPES, score_days, summarise
from .evaluate import aggregate, baseline_accuracy, synthetic_tests
from .ingest import load_entities, load_indoor, load_weather
from .quality import assess, drop_duplicates, quality_events
from .sources import DataSource

log = logging.getLogger("waste_intel")
STATUS = OUT_DIR / "status.json"


def _setup_logging():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.FileHandler(OUT_DIR / "service2.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.propagate = False


def _status(run_id: str, step: str, done: int = 0, total: int = 0, **extra):
    STATUS.write_text(json.dumps({"run_id": run_id, "step": step, "done": done, "total": total,
                                  "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"), **extra},
                                 ensure_ascii=False, indent=1), encoding="utf-8")


def _data_end(entities: list[dict], source: str, today: date) -> date | None:
    """Last day on which most of a source's meters reported a full day, so an outage on
    our side (or yesterday's incomplete day) is not mistaken for broken meters."""
    counts: dict = {}
    for e in entities:
        if e["source"] != source:
            continue
        full = e["matrix"].notna().sum(axis=1) >= CONFIG["day_min_slots"]
        for d in full[full].index:
            if d < today:
                counts[d] = counts.get(d, 0) + 1
    if not counts:
        return None
    last = max(counts)
    peak = max(v for d, v in counts.items() if d > last - timedelta(days=30))
    for d in sorted(counts, reverse=True):
        if counts[d] >= 0.6 * peak:
            return d
    return last


def _outage_days(entities: list[dict], source: str, end: date, days: int = 60) -> set:
    """Days in the last `days` on which fewer than half of a source's usual meters reported."""
    counts: dict = {}
    for e in entities:
        if e["source"] == source:
            full = e["matrix"].notna().sum(axis=1) >= CONFIG["day_min_slots"]
            for d in full[full].index:
                counts[d] = counts.get(d, 0) + 1
    window = [end - timedelta(days=i) for i in range(days)]
    typical = float(np.median([counts.get(d, 0) for d in window])) if counts else 0
    return {d for d in window if counts.get(d, 0) < 0.5 * typical}


def _clean(o):
    """NaN and infinity are not valid JSON: the API would fail on them."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _json(obj) -> str:
    obj = _clean(obj)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        return str(o)
    return json.dumps(_clean(json.loads(json.dumps(obj, default=default))), ensure_ascii=False, indent=1,
                      allow_nan=False)


def run(data: DataSource, skip_eval: bool = False, only: tuple[str, ...] = ("sensor", "eunice", "dei")):
    _setup_logging()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    today = date.today()
    log.info("run %s started (%s)", run_id, MODEL_VERSION)

    # 1. ingestion
    _status(run_id, "1/6 φόρτωση δεδομένων")
    entities, fallback_price = load_entities(data)
    weather = load_weather(data)
    indoor = load_indoor(data, [e["thermo_id"] for e in entities if e["thermo_id"]])
    sources = set(only)
    entities = [e for e in entities if e["source"] in sources]
    log.info("ingest done: %d meters, weather to %s, indoor temperature for %d schools, fallback price %.3f €/kWh",
             len(entities), weather[~weather["clim"]].index.max(), len(indoor), fallback_price)

    # 2. quality control
    _status(run_id, "2/6 έλεγχος ποιότητας", 0, len(entities))
    ends = {s: _data_end(entities, s, today) for s in sources}
    log.info("data end per source: %s", {k: str(v) for k, v in ends.items()})
    outages = {s: _outage_days(entities, s, ends[s]) for s in sources if ends[s]}
    for s, days in outages.items():
        if days:
            log.info("collection outage (%s): %s", s, ", ".join(str(d) for d in sorted(days)))
    for e in entities:
        end = ends[e["source"]]
        e["end"] = end
        ref = end if e["source"] == "dei" else today
        e["qc"] = assess(e, end, ref) if end else {"ok": False, "reasons": ["χωρίς δεδομένα"], "metrics": {}}
    drop_duplicates(entities)
    included = [e for e in entities if e["qc"]["ok"]]
    for s in sorted(sources):
        n = sum(e["source"] == s for e in entities)
        k = sum(e["source"] == s for e in included)
        log.info("quality: %s %d of %d meters included", s, k, n)
    (run_dir / "quality.json").write_text(_json([
        {"id": e["id"], "source": e["source"], "name": e["name"], "building_id": e["building_id"],
         "included": e["qc"]["ok"], "reasons": e["qc"]["reasons"], **e["qc"]["metrics"]} for e in entities]),
        encoding="utf-8")

    # 3-4. baseline, scoring, waste quantification
    results = []
    window = 2 * CONFIG["monitor_days"]
    for i, e in enumerate(included, 1):
        _status(run_id, "3/6 βάση αναφοράς και ανίχνευση", i - 1, len(included), current=e["name"])
        t1 = time.time()
        end = e["end"]
        start = end - timedelta(days=window - 1)
        b = Baseline(e["matrix"], weather, e["school"])
        ind = indoor.get(e["thermo_id"]) if e["thermo_id"] else None
        scored = score_days(b, start, end, e["school"], weather, ind)
        qev = quality_events(e, end - timedelta(days=CONFIG["monitor_days"] - 1), end, outages.get(e["source"]))
        summary = summarise(e, scored, qev, end)
        acc = baseline_accuracy(b, end - timedelta(days=CONFIG["monitor_days"] - 1), end)
        results.append({
            "id": e["id"], "source": e["source"], "name": e["name"], "building_id": e["building_id"],
            "device": e["device"], "school": e["school"], "pv_net": e["pv"],
            "price_eur_kwh": round(e["price"], 4), "price_source": e["price_source"],
            "data_end": end, "historical": e["source"] == "dei",
            "history_days": e["qc"]["metrics"].get("history_days"),
            "accuracy": acc, "days": scored["days"], **summary,
        })
        n_reg = sum(1 for x in summary["detections"] if x.get("registered"))
        log.info("[%d/%d] %s (%s): status=%s trend=%s registered=%d annual=%s € (%.1fs)", i, len(included),
                 e["name"], e["id"], summary["status"], summary["trend"], n_reg, f"{summary['annual_eur']:,}",
                 time.time() - t1)

    # 5. evaluation
    evaluation = {"baseline": _accuracy_summary(results)}
    if not skip_eval:
        tests = []
        for i, e in enumerate(included, 1):
            _status(run_id, "4/6 αξιολόγηση με τεχνητά περιστατικά", i - 1, len(included), current=e["name"])
            t1 = time.time()
            try:
                ts = synthetic_tests(e, weather, e["end"])
            except Exception as exc:  # one meter must not stop the run
                log.warning("evaluation failed for %s: %s", e["id"], exc)
                continue
            tests.append(ts)
            log.info("[eval %d/%d] %s: %s (%.1fs)", i, len(included), e["id"],
                     ", ".join(f"{t['kind']}={'✓' if t.get('detected') else ('✗' if 'detected' in t else t['flags'])}" for t in ts),
                     time.time() - t1)
        evaluation["synthetic"] = aggregate(tests)
        evaluation["synthetic_by_source"] = {
            s: aggregate([ts for ts, e in zip(tests, included) if e["source"] == s]) for s in sorted(sources)}
    log.info("evaluation: %s", json.dumps(evaluation.get("synthetic", {}), ensure_ascii=False))

    # 6. outputs
    _status(run_id, "5/6 αποθήκευση αποτελεσμάτων")
    register = sorted(
        [{"id": r["id"], "name": r["name"], "source": r["source"], "historical": r["historical"], **line}
         for r in results for line in r["register"]],
        key=lambda x: -x["annual_eur"])
    config = {"run_id": run_id, "model_version": MODEL_VERSION, "run_date": today, "data_end": ends,
              "fallback_price_eur_kwh": round(fallback_price, 4), "thresholds": CONFIG,
              "elapsed_s": round(time.time() - t0)}
    files = {"config.json": config, "buildings.json": results, "register.json": register,
             "evaluation.json": evaluation}
    for name, obj in files.items():
        (run_dir / name).write_text(_json(obj), encoding="utf-8")
    from .report import build_report
    _status(run_id, "6/6 report")
    report = build_report(config, entities, results, register, evaluation)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    latest = OUT_DIR / "latest"
    latest.mkdir(exist_ok=True)
    for name in list(files) + ["quality.json", "report.md"]:
        shutil.copy2(run_dir / name, latest / name)
    _status(run_id, "ολοκληρώθηκε", len(included), len(included), elapsed_s=round(time.time() - t0))
    log.info("run %s finished in %.0fs: %d meters, %d register lines, %s € a year", run_id, time.time() - t0,
             len(results), len(register), f"{sum(x['annual_eur'] for x in register):,}")


def _accuracy_summary(results: list[dict]) -> dict:
    out = {}
    for s in sorted({r["source"] for r in results}):
        acc = [r["accuracy"] for r in results if r["source"] == s and r["accuracy"]]
        if not acc:
            continue
        better = sum(a["daily"]["rmse"] < a["daily_reference"]["rmse"] for a in acc)
        rel = [a["daily"]["rmse"] / a["mean_daily_kwh"] for a in acc if a["mean_daily_kwh"] > 0]
        rel_ref = [a["daily_reference"]["rmse"] / a["mean_daily_kwh"] for a in acc if a["mean_daily_kwh"] > 0]
        out[s] = {"meters": len(acc), "better_than_reference": better,
                  "median_daily_rmse_share": round(float(np.median(rel)), 3),
                  "median_daily_rmse_share_reference": round(float(np.median(rel_ref)), 3)}
    return out
