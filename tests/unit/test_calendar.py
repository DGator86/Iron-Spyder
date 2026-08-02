"""Market calendar tests.

The dates below are the published NYSE schedule, not values read back from the
implementation.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from spy_der.runtime import calendar as cal

# Published NYSE full-day closures.
HOLIDAYS_2026 = {
    date(2026, 1, 1),  # New Year's Day (Thu)
    date(2026, 1, 19),  # MLK
    date(2026, 2, 16),  # Presidents Day
    date(2026, 4, 3),  # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 6, 19),  # Juneteenth (Fri)
    date(2026, 7, 3),  # Independence Day observed (4th is a Saturday)
    date(2026, 9, 7),  # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas (Fri)
}

HOLIDAYS_2025 = {
    date(2025, 1, 1),
    date(2025, 1, 20),  # MLK
    date(2025, 2, 17),
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),
    date(2025, 6, 19),
    date(2025, 7, 4),
    date(2025, 9, 1),
    date(2025, 11, 27),
    date(2025, 12, 25),
}


def test_2026_holidays_match_the_published_schedule():
    assert cal.market_holidays(2026) == HOLIDAYS_2026


def test_2025_holidays_match_the_published_schedule():
    assert cal.market_holidays(2025) == HOLIDAYS_2025


def test_good_friday_tracks_easter():
    # Easter Sunday dates are fixed astronomy/convention, not a lookup table.
    assert cal.easter_sunday(2025) == date(2025, 4, 20)
    assert cal.easter_sunday(2026) == date(2026, 4, 5)
    assert cal.easter_sunday(2027) == date(2027, 3, 28)
    assert date(2027, 3, 26) in cal.market_holidays(2027)  # Good Friday 2027


def test_juneteenth_only_from_2022():
    assert date(2021, 6, 18) not in cal.market_holidays(2021)
    assert date(2022, 6, 20) in cal.market_holidays(2022)  # 19th was a Sunday


def test_new_year_on_saturday_is_not_observed_on_the_prior_december():
    """The NYSE does not close 31 December for a Saturday 1 January."""
    assert date(2028, 1, 1).weekday() == 5
    assert cal.is_trading_day(date(2027, 12, 31))
    assert date(2027, 12, 31) not in cal.market_holidays(2027)


def test_holiday_on_sunday_is_observed_on_monday():
    # 4 July 2027 is a Sunday, observed Monday the 5th.
    assert date(2027, 7, 4).weekday() == 6
    assert date(2027, 7, 5) in cal.market_holidays(2027)


def test_weekends_are_not_trading_days():
    assert not cal.is_trading_day(date(2026, 8, 1))  # Saturday
    assert not cal.is_trading_day(date(2026, 8, 2))  # Sunday
    assert cal.is_trading_day(date(2026, 8, 3))  # Monday


def test_early_closes_2026():
    assert cal.early_closes(2026) == {date(2026, 11, 27), date(2026, 12, 24)}
    assert cal.is_early_close(date(2026, 11, 27))
    assert cal.close_time(date(2026, 11, 27)) == cal.EARLY_CLOSE
    assert cal.close_time(date(2026, 8, 3)) == cal.REGULAR_CLOSE


def test_july_third_is_an_early_close_only_when_it_trades():
    # 2024: 4 July was a Thursday, so the 3rd was a shortened session.
    assert date(2024, 7, 3) in cal.early_closes(2024)
    # 2026: 4 July is a Saturday, so the 3rd is a full closure, not a half day.
    assert date(2026, 7, 3) in cal.market_holidays(2026)
    assert date(2026, 7, 3) not in cal.early_closes(2026)


def test_session_durations():
    regular = cal.session_for(date(2026, 8, 3))
    assert regular is not None
    assert regular.duration_minutes == pytest.approx(390.0)
    assert not regular.is_early_close

    early = cal.session_for(date(2026, 11, 27))
    assert early is not None
    assert early.duration_minutes == pytest.approx(210.0)
    assert early.is_early_close

    assert cal.session_for(date(2026, 12, 25)) is None


def test_session_open_handles_daylight_saving():
    """09:30 ET is 13:30 UTC in summer and 14:30 UTC in winter."""
    summer = cal.session_for(date(2026, 8, 3))
    winter = cal.session_for(date(2026, 1, 5))
    assert summer is not None and winter is not None
    assert summer.open_at.hour == 13
    assert winter.open_at.hour == 14
    # Both are still 390-minute sessions.
    assert summer.duration_minutes == winter.duration_minutes == pytest.approx(390.0)


@pytest.mark.parametrize(
    "moment,expected_open",
    [
        (datetime(2026, 8, 3, 13, 29, tzinfo=UTC), False),  # one minute early
        (datetime(2026, 8, 3, 13, 30, tzinfo=UTC), True),  # the open
        (datetime(2026, 8, 3, 19, 59, tzinfo=UTC), True),
        (datetime(2026, 8, 3, 20, 0, tzinfo=UTC), False),  # the close is exclusive
        (datetime(2026, 8, 1, 15, 0, tzinfo=UTC), False),  # Saturday
        (datetime(2026, 12, 25, 15, 0, tzinfo=UTC), False),  # Christmas
    ],
)
def test_is_market_open(moment, expected_open):
    assert cal.is_market_open(moment) is expected_open


def test_minutes_to_close_counts_down_and_floors_at_zero():
    assert cal.minutes_to_close(datetime(2026, 8, 3, 13, 30, tzinfo=UTC)) == pytest.approx(390.0)
    assert cal.minutes_to_close(datetime(2026, 8, 3, 19, 30, tzinfo=UTC)) == pytest.approx(30.0)
    assert cal.minutes_to_close(datetime(2026, 8, 3, 20, 0, tzinfo=UTC)) == 0.0
    assert cal.minutes_to_close(datetime(2026, 8, 1, 15, 0, tzinfo=UTC)) == 0.0


def test_minutes_to_close_respects_an_early_close():
    """A half day must not report time remaining against a 16:00 close."""
    moment = datetime(2026, 11, 27, 17, 30, tzinfo=UTC)  # 12:30 ET
    assert cal.minutes_to_close(moment) == pytest.approx(30.0)


def test_next_session_skips_weekends_and_holidays():
    friday_evening = datetime(2026, 8, 7, 21, 0, tzinfo=UTC)
    following = cal.next_session(friday_evening)
    assert following is not None
    assert following.day == date(2026, 8, 10)  # Monday

    before_christmas = datetime(2026, 12, 24, 19, 0, tzinfo=UTC)
    following = cal.next_session(before_christmas)
    assert following is not None
    assert following.day == date(2026, 12, 28)  # Christmas Fri, then the weekend


def test_naive_datetimes_are_treated_as_utc():
    aware = datetime(2026, 8, 3, 15, 0, tzinfo=UTC)
    naive = datetime(2026, 8, 3, 15, 0)
    assert cal.is_market_open(naive) == cal.is_market_open(aware)
    assert cal.minutes_to_close(naive) == pytest.approx(cal.minutes_to_close(aware))


def test_sessions_between_counts_trading_days():
    sessions = cal.sessions_between(date(2026, 8, 3), date(2026, 8, 9))
    assert [s.day.weekday() for s in sessions] == [0, 1, 2, 3, 4]


def test_decision_times_stay_inside_the_session():
    session = cal.session_for(date(2026, 8, 3))
    assert session is not None
    times = cal.decision_times(session, timedelta(minutes=30))
    assert times[0] == session.open_at
    assert all(t < session.close_at for t in times)
    assert len(times) == 13  # 390 / 30

    early = cal.session_for(date(2026, 11, 27))
    assert early is not None
    assert len(cal.decision_times(early, timedelta(minutes=30))) == 7  # 210 / 30


def test_decision_times_reject_a_non_positive_interval():
    session = cal.session_for(date(2026, 8, 3))
    assert session is not None
    with pytest.raises(ValueError, match="positive"):
        cal.decision_times(session, timedelta(0))
