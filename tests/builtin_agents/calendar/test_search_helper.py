"""Unit tests for calendar search_helper time-window parsing.

These cover the contract introduced after we removed ``parse_time_range``:
the LLM now passes explicit ISO-8601 ``time_min``/``time_max`` and there
is **no fallback** for missing or invalid input — every problem must
raise ``ValueError`` so the ReAct loop can surface it to the LLM.
"""

from datetime import datetime, timezone

import pytest

from koa.builtin_agents.calendar.search_helper import (
    parse_iso_datetime,
    resolve_time_window,
)


class TestParseIsoDatetime:
    def test_full_iso_with_offset_preserves_offset(self):
        result = parse_iso_datetime("2026-04-27T00:00:00-07:00", field_name="time_min")
        assert result.utcoffset().total_seconds() == -7 * 3600
        assert result.year == 2026 and result.month == 4 and result.day == 27

    def test_zulu_z_suffix_is_treated_as_utc(self):
        result = parse_iso_datetime("2026-04-27T07:00:00Z", field_name="time_min")
        assert result.tzinfo is not None
        assert result.utcoffset().total_seconds() == 0

    def test_naive_datetime_is_localised_to_user_tz(self):
        result = parse_iso_datetime(
            "2026-04-27T09:00:00",
            field_name="time_min",
            user_tz="America/Los_Angeles",
        )
        assert result.tzinfo is not None
        # PDT in late April is UTC-7
        assert result.utcoffset().total_seconds() == -7 * 3600

    def test_date_only_is_midnight_in_user_tz(self):
        result = parse_iso_datetime(
            "2026-04-27",
            field_name="time_min",
            user_tz="America/Los_Angeles",
        )
        assert (result.hour, result.minute, result.second) == (0, 0, 0)
        assert result.utcoffset().total_seconds() == -7 * 3600

    def test_date_only_without_tz_falls_back_to_utc(self):
        result = parse_iso_datetime("2026-04-27", field_name="time_min")
        assert result.tzinfo == timezone.utc
        assert result.hour == 0

    def test_unknown_user_tz_falls_back_to_utc(self):
        result = parse_iso_datetime(
            "2026-04-27T09:00:00",
            field_name="time_min",
            user_tz="Mars/Phobos",
        )
        assert result.tzinfo == timezone.utc

    def test_existing_datetime_passes_through(self):
        dt = datetime(2026, 4, 27, 9, 0, tzinfo=timezone.utc)
        assert parse_iso_datetime(dt, field_name="time_min") is dt

    def test_naive_datetime_object_is_localised(self):
        dt = datetime(2026, 4, 27, 9, 0)
        result = parse_iso_datetime(dt, field_name="time_min", user_tz="America/Los_Angeles")
        assert result.tzinfo is not None
        assert result.utcoffset().total_seconds() == -7 * 3600

    @pytest.mark.parametrize("missing", [None, "", "   "])
    def test_missing_value_raises_with_field_name(self, missing):
        with pytest.raises(ValueError, match="time_min is required"):
            parse_iso_datetime(missing, field_name="time_min")

    def test_garbage_string_raises_with_field_name(self):
        with pytest.raises(ValueError, match="time_max must be ISO-8601"):
            parse_iso_datetime("last week", field_name="time_max")

    def test_partial_garbage_raises(self):
        with pytest.raises(ValueError, match="time_min must be ISO-8601"):
            parse_iso_datetime("2026/04/27", field_name="time_min")

    def test_non_string_non_datetime_raises(self):
        with pytest.raises(ValueError, match="time_min must be a string"):
            parse_iso_datetime(12345, field_name="time_min")


class TestResolveTimeWindow:
    def test_happy_path_returns_parsed_pair(self):
        a, b = resolve_time_window("2026-04-27", "2026-05-04", user_tz="America/Los_Angeles")
        assert a < b
        assert a.day == 27 and b.day == 4

    def test_full_iso_pair(self):
        a, b = resolve_time_window("2026-04-27T00:00:00-07:00", "2026-05-04T00:00:00-07:00")
        assert (b - a).days == 7

    def test_equal_bounds_raises(self):
        with pytest.raises(ValueError, match="strictly after"):
            resolve_time_window("2026-04-27", "2026-04-27")

    def test_inverted_bounds_raises(self):
        with pytest.raises(ValueError, match="strictly after"):
            resolve_time_window("2026-05-04", "2026-04-27")

    def test_missing_min_raises_field_scoped(self):
        with pytest.raises(ValueError, match="time_min is required"):
            resolve_time_window(None, "2026-05-04")

    def test_missing_max_raises_field_scoped(self):
        with pytest.raises(ValueError, match="time_max is required"):
            resolve_time_window("2026-04-27", "")

    def test_invalid_min_raises_field_scoped(self):
        with pytest.raises(ValueError, match="time_min must be ISO-8601"):
            resolve_time_window("last week", "2026-05-04")

    def test_invalid_max_raises_field_scoped(self):
        with pytest.raises(ValueError, match="time_max must be ISO-8601"):
            resolve_time_window("2026-04-27", "next monday")
