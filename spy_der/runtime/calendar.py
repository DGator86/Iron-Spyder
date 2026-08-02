"""US equity market calendar (NYSE/NASDAQ rules).

An unattended runner has to know, without being told, whether the market is
open right now, when it closes today, and when to wake next. Getting this wrong
is not cosmetic: ``minutes_to_close`` drives the end-of-session exit, the
late-session entry block, and the assignment cutoff, so a runner that thinks a
half day ends at 16:00 will hold positions into a close that already happened.

Rules are computed rather than tabulated, so the calendar does not expire at
the end of a hard-coded year. Everything is derived from stdlib
:mod:`zoneinfo`; no market-calendar dependency is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)

#: Juneteenth became a market holiday in 2022.
JUNETEENTH_FIRST_YEAR = 2022


def easter_sunday(year: int) -> date:
    """Gregorian Easter via the Meeus/Jones/Butcher algorithm.

    Needed because Good Friday is the one NYSE holiday with no fixed date and
    no weekday rule.
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lunar = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lunar) // 451
    month, day = divmod(h + lunar - 7 * m + 114, 31)
    return date(year, month, day + 1)


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The ``n``-th ``weekday`` of a month (Monday is 0)."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def last_weekday(year: int, month: int, weekday: int) -> date:
    """The final ``weekday`` of a month."""
    following = date(year + (month == 12), (month % 12) + 1, 1)
    last = following - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date) -> date | None:
    """Apply the weekend-observation rule to a fixed-date holiday.

    Saturday shifts back to Friday, Sunday forward to Monday — except New
    Year's Day, which the NYSE does not observe on the preceding 31 December
    because that falls in the prior year.
    """
    if day.weekday() == 5:  # Saturday
        if day.month == 1 and day.day == 1:
            return None
        return day - timedelta(days=1)
    if day.weekday() == 6:  # Sunday
        return day + timedelta(days=1)
    return day


@lru_cache(maxsize=64)
def market_holidays(year: int) -> frozenset[date]:
    """Full-day NYSE closures for ``year``."""
    days: set[date] = set()

    for fixed in (
        date(year, 1, 1),  # New Year's Day
        date(year, 7, 4),  # Independence Day
        date(year, 12, 25),  # Christmas
    ):
        observed = _observed(fixed)
        if observed is not None:
            days.add(observed)

    if year >= JUNETEENTH_FIRST_YEAR:
        observed = _observed(date(year, 6, 19))
        if observed is not None:
            days.add(observed)

    days.add(nth_weekday(year, 1, 0, 3))  # MLK Day
    days.add(nth_weekday(year, 2, 0, 3))  # Presidents Day
    days.add(last_weekday(year, 5, 0))  # Memorial Day
    days.add(nth_weekday(year, 9, 0, 1))  # Labor Day
    days.add(nth_weekday(year, 11, 3, 4))  # Thanksgiving
    days.add(easter_sunday(year) - timedelta(days=2))  # Good Friday

    return frozenset(days)


@lru_cache(maxsize=64)
def early_closes(year: int) -> frozenset[date]:
    """Days closing at 13:00 ET."""
    days: set[date] = set()
    holidays = market_holidays(year)

    # Day after Thanksgiving.
    day_after = nth_weekday(year, 11, 3, 4) + timedelta(days=1)
    if day_after not in holidays:
        days.add(day_after)

    # 3 July, when it is a weekday and Independence Day itself is observed on
    # the 4th. If the 4th lands on a weekend the shortened session does not
    # apply.
    july_third = date(year, 7, 3)
    if (
        july_third.weekday() < 5
        and date(year, 7, 4).weekday() < 5
        and july_third not in holidays
    ):
        days.add(july_third)

    # Christmas Eve, when it is a weekday and Christmas is observed on the 25th.
    christmas_eve = date(year, 12, 24)
    if (
        christmas_eve.weekday() < 5
        and date(year, 12, 25).weekday() < 5
        and christmas_eve not in holidays
    ):
        days.add(christmas_eve)

    return frozenset(days)


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in market_holidays(day.year)


def is_early_close(day: date) -> bool:
    return is_trading_day(day) and day in early_closes(day.year)


def close_time(day: date) -> time:
    return EARLY_CLOSE if is_early_close(day) else REGULAR_CLOSE


@dataclass(frozen=True)
class Session:
    """One trading session, in UTC."""

    day: date
    open_at: datetime
    close_at: datetime
    is_early_close: bool

    @property
    def duration_minutes(self) -> float:
        return (self.close_at - self.open_at).total_seconds() / 60.0

    def contains(self, moment: datetime) -> bool:
        return self.open_at <= moment < self.close_at

    def minutes_to_close(self, moment: datetime) -> float:
        """Minutes remaining, clamped to the session; zero once closed."""
        if moment >= self.close_at:
            return 0.0
        if moment <= self.open_at:
            return self.duration_minutes
        return (self.close_at - moment).total_seconds() / 60.0


def session_for(day: date) -> Session | None:
    """The session on ``day``, or ``None`` when the market is closed."""
    if not is_trading_day(day):
        return None
    early = is_early_close(day)
    return Session(
        day=day,
        open_at=datetime.combine(day, REGULAR_OPEN, tzinfo=EASTERN).astimezone(UTC),
        close_at=datetime.combine(
            day, EARLY_CLOSE if early else REGULAR_CLOSE, tzinfo=EASTERN
        ).astimezone(UTC),
        is_early_close=early,
    )


def _as_utc(moment: datetime) -> datetime:
    """Interpret a naive datetime as UTC; convert an aware one."""
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def current_session(moment: datetime) -> Session | None:
    """The session in progress at ``moment``, or ``None`` outside hours.

    Checks the Eastern-local day, and also the previous one: a session that
    opens at 09:30 ET spans a UTC date boundary during part of the year, so
    keying only off the UTC date would miss it.
    """
    utc_moment = _as_utc(moment)
    eastern_day = utc_moment.astimezone(EASTERN).date()
    for candidate in (eastern_day, eastern_day - timedelta(days=1)):
        session = session_for(candidate)
        if session is not None and session.contains(utc_moment):
            return session
    return None


def is_market_open(moment: datetime) -> bool:
    return current_session(moment) is not None


def next_session(moment: datetime, *, limit_days: int = 30) -> Session | None:
    """The next session starting strictly after ``moment``."""
    utc_moment = _as_utc(moment)
    day = utc_moment.astimezone(EASTERN).date()
    for offset in range(limit_days + 1):
        session = session_for(day + timedelta(days=offset))
        if session is not None and session.open_at > utc_moment:
            return session
    return None


def minutes_to_close(moment: datetime) -> float:
    """Minutes left in the current session; zero when the market is closed."""
    session = current_session(moment)
    return session.minutes_to_close(_as_utc(moment)) if session else 0.0


def sessions_between(start: date, end: date) -> list[Session]:
    """Every session in ``[start, end]``, inclusive."""
    out: list[Session] = []
    day = start
    while day <= end:
        session = session_for(day)
        if session is not None:
            out.append(session)
        day += timedelta(days=1)
    return out


def decision_times(session: Session, interval: timedelta) -> list[datetime]:
    """Decision timestamps across a session at a fixed interval.

    The close itself is excluded: a decision taken at the bell cannot be acted
    on, and the position manager already forces closure before then.
    """
    if interval <= timedelta(0):
        raise ValueError("interval must be positive")
    out: list[datetime] = []
    moment = session.open_at
    while moment < session.close_at:
        out.append(moment)
        moment += interval
    return out
