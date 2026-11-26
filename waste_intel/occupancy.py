"""Occupancy calendar: Greek public holidays and school closures.

School closures follow the Ministry's usual pattern (Christmas 24/12-7/1, Easter from Holy
Monday to Thomas Sunday, summer from 21/6 to 10/9). Exact dates vary slightly per year and
level, so these are an approximation and are listed in every report.
"""

from datetime import date, timedelta
from functools import lru_cache


def orthodox_easter(year: int) -> date:
    a, b, c = year % 19, year % 4, year % 7
    d = (19 * a + 15) % 30
    e = (2 * b + 4 * c + 6 * d + 6) % 7
    return date(year, 3, 22) + timedelta(days=d + e + 13)


@lru_cache(maxsize=None)
def public_holidays(year: int) -> frozenset:
    e = orthodox_easter(year)
    fixed = [(1, 1), (1, 6), (3, 25), (5, 1), (8, 15), (10, 28), (12, 25), (12, 26)]
    days = {date(year, m, d) for m, d in fixed}
    days |= {e - timedelta(days=48), e - timedelta(days=2), e, e + timedelta(days=1), e + timedelta(days=50)}
    return frozenset(days)


@lru_cache(maxsize=None)
def school_closures(year: int) -> frozenset:
    e = orthodox_easter(year)
    days = set(public_holidays(year))
    days |= {date(year, 1, 30), date(year, 11, 17)}                       # Three Hierarchs, Polytechnic
    days |= {e - timedelta(days=6) + timedelta(days=i) for i in range(14)}  # Holy Monday .. Thomas Sunday
    d = date(year, 6, 21)
    while d <= date(year, 9, 10):
        days.add(d)
        d += timedelta(days=1)
    for d0, d1 in ((date(year, 1, 1), date(year, 1, 7)), (date(year, 12, 24), date(year, 12, 31))):
        d = d0
        while d <= d1:
            days.add(d)
            d += timedelta(days=1)
    return frozenset(days)


def is_holiday(d: date) -> bool:
    return d in public_holidays(d.year)


def is_open(d: date, school: bool) -> bool:
    if d.weekday() >= 5:
        return False
    return d not in (school_closures(d.year) if school else public_holidays(d.year))


def day_type(d: date, school: bool) -> str:
    """Baseline stratum. Schools get a separate stratum for weekday closures (vacations)."""
    if is_holiday(d) or d.weekday() == 6:
        return "sunday_holiday"
    if d.weekday() == 5:
        return "saturday"
    if school and not is_open(d, True):
        return "vacation_weekday"
    return "workday"
