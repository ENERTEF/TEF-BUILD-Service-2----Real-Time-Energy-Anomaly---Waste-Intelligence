"""The data the pipeline needs, and nothing about where it is stored.

A deployment implements `DataSource` once, against its own database, files or APIs, and
passes it to `run.run()`. Everything downstream works on the structures defined here.

Meter sources (the `source` field of a `Meter`)
  sensor  school sub-meters
  eunice  municipal smart meters (EUNICE)
  dei     utility telemetry per supply number (DEI); historical, so quality is judged
          against its own last reading rather than today
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Protocol

import pandas as pd


@dataclass
class Meter:
    id: str                       # unique across sources, e.g. "sensor-<device>"
    source: str                   # "sensor", "eunice" or "dei"
    device: str                   # the meter's own identifier within its source
    building_id: int | None       # building it belongs to; prices and PV are joined on it
    name: str                     # display name used in reports
    school: bool = False          # schools follow the school calendar and opening hours
    thermo_id: str | None = None  # key into indoor_temperature(), where a thermostat exists
    shared_with: int = 0          # number of other buildings fed by the same physical meter


class DataSource(Protocol):
    def meters(self) -> Iterable[tuple[Meter, pd.DataFrame]]:
        """Every candidate meter with its 15-minute readings.

        Readings: columns `t` (interval start, epoch seconds UTC) and `kwh` (energy in the
        interval). Missing intervals are simply absent.
        """

    def pv_generation(self) -> dict[int, pd.DataFrame]:
        """Inverter generation per building_id, same columns as meter readings. Added back
        to grid import so that a sunny day does not look like a drop in demand."""

    def bill_totals(self, year_from: int, year_to: int) -> Iterable[tuple[int, float, float]]:
        """(building_id, € excluding municipal fees, kWh) of electricity bills in the years
        given, without correction or extraordinary bills. Gives each building its €/kWh."""

    def weather_daily(self) -> pd.DataFrame:
        """One row per day: `day` (date), `t_mean` (°C), `hdd` (heating degree days, base
        18 °C), `cdd` (cooling degree days, base 22 °C), `clim` (True where the row is
        climatology rather than an observation)."""

    def indoor_temperature(self, thermo_ids: list[str]) -> Iterable[tuple[str, date, float]]:
        """(thermo_id, day, mean indoor °C) from thermostats. Used only to corroborate
        heating or cooling in a closed building."""
