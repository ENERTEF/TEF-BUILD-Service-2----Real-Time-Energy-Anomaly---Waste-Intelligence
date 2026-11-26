"""Data ingestion: every 15-minute electricity series, as one matrix per meter
(rows = local calendar days, columns = 96 intervals, kWh, NaN where nothing arrived),
with the building's €/kWh, the weather and indoor temperatures.

For a building with PV, consumption is grid import plus inverter generation, so a sunny
day does not look like a drop in demand.

All reads go through a `DataSource` (see `sources.py`).
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import CONFIG, SLOTS, TZ
from .sources import DataSource

log = logging.getLogger("waste_intel")


def _matrix(df: pd.DataFrame) -> pd.DataFrame:
    """df has columns t (epoch seconds, interval start) and kwh."""
    ts = pd.to_datetime(df["t"], unit="s", utc=True).dt.tz_convert(TZ)
    slot = ts.dt.hour * 4 + ts.dt.minute // 15
    m = (pd.DataFrame({"day": ts.dt.date, "slot": slot, "kwh": df["kwh"].astype(float)})
         .pivot_table(index="day", columns="slot", values="kwh", aggfunc="sum"))
    return m.reindex(columns=range(SLOTS)).sort_index()


def _prices(data: DataSource) -> tuple[dict, float]:
    """€/kWh per building from its bills, inside a sanity band; the portfolio median
    for buildings without usable bills."""
    y0, y1 = CONFIG["price_years"]
    prices = {}
    for bid, eur, kwh in data.bill_totals(y0, y1):
        if kwh and eur:
            p = float(eur) / float(kwh)
            if CONFIG["price_min"] <= p <= CONFIG["price_max"]:
                prices[bid] = p
    fallback = float(np.median(list(prices.values()))) if prices else 0.20
    return prices, fallback


def load_weather(data: DataSource) -> pd.DataFrame:
    return data.weather_daily().set_index("day").sort_index()


def load_entities(data: DataSource) -> tuple[list[dict], float]:
    """All candidate meters, before quality control."""
    prices, fallback = _prices(data)
    pv = {bid: _matrix(g) for bid, g in data.pv_generation().items()}

    entities: list[dict] = []
    for meter, readings in data.meters():
        m = _matrix(readings)
        has_pv = meter.building_id in pv
        if has_pv:
            m = m + pv[meter.building_id].reindex(index=m.index, columns=m.columns).fillna(0.0)
        entities.append({
            "id": meter.id, "source": meter.source, "device": meter.device,
            "building_id": meter.building_id, "name": meter.name, "school": meter.school,
            "thermo_id": meter.thermo_id, "shared_with": meter.shared_with,
            "matrix": m, "pv": has_pv,
            "price": prices.get(meter.building_id, fallback),
            "price_source": "bills" if meter.building_id in prices else "portfolio median",
        })
    for s in ("sensor", "eunice", "dei"):
        log.info("ingest: %d %s meters", sum(e["source"] == s for e in entities), s)
    return entities, fallback


def load_indoor(data: DataSource, thermo_ids: list[str]) -> dict:
    """Daily indoor temperature per thermostat id, for corroboration only."""
    if not thermo_ids:
        return {}
    out = defaultdict(dict)
    for tid, day, t in data.indoor_temperature(list(set(thermo_ids))):
        if t is not None:
            out[tid][day] = float(t)
    return out
